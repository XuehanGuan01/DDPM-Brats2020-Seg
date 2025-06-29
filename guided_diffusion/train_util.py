import copy
import functools
import os
import csv

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# 移除Visdom相关导入和变量

INITIAL_LOG_LOSS_SCALE = 20.0  # 初始对数损失缩放值，适用于ImageNet实验

def visualize(img):
    """将图像归一化到[0,1]范围以便可视化"""
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        metrics_file="training_metrics.csv",  # 新增参数：CSV文件路径
    ):
        """初始化训练循环"""
        self.model = model  # 模型
        self.dataloader = dataloader  # 数据加载器
        self.classifier = classifier  # 分类器（如果使用）
        self.diffusion = diffusion  # 扩散过程
        self.data = data  # 数据集
        self.batch_size = batch_size  # 批次大小
        self.microbatch = microbatch if microbatch > 0 else batch_size  # 微批次大小
        self.lr = lr  # 学习率
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )  # 指数移动平均（EMA）衰减率
        self.log_interval = log_interval  # 日志记录间隔
        self.save_interval = save_interval  # 模型保存间隔
        self.resume_checkpoint = resume_checkpoint  # 恢复检查点路径
        self.use_fp16 = use_fp16  # 是否使用混合精度训练
        self.fp16_scale_growth = fp16_scale_growth  # 混合精度缩放增长率
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)  # 时间步采样器
        self.weight_decay = weight_decay  # 权重衰减
        self.lr_anneal_steps = lr_anneal_steps  # 学习率退火步数
        self.metrics_file = metrics_file  # CSV文件路径

        self.step = 0  # 当前步数
        self.resume_step = 0  # 恢复步数
        self.global_batch = self.batch_size * dist.get_world_size()  # 全局批次大小

        self.sync_cuda = th.cuda.is_available()  # 是否支持CUDA

        self._load_and_sync_parameters()  # 加载和同步模型参数
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )  # 混合精度训练器

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )  # 优化器
        if self.resume_step:
            self._load_optimizer_state()  # 加载优化器状态
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]  # 加载EMA参数
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]  # 初始化EMA参数

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )  # 使用分布式数据并行
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "分布式训练需要CUDA支持，梯度同步可能不正确！"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # 初始化CSV文件，写入表头
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Loss'])  # 如果需要其他指标，可添加

    def _load_and_sync_parameters(self):
        """加载并同步模型参数"""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('恢复模型')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"从检查点加载模型: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        """加载EMA参数"""
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"从检查点加载EMA: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        """加载优化器状态"""
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"从检查点加载优化器状态: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        """运行训练循环"""
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            try:
                batch, cond = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch, cond = next(data_iter)

            self.run_step(batch, cond)

            i += 1

            if self.step % self.log_interval == 0:
                logger.dumpkvs()  # 记录日志
            if self.step % self.save_interval == 0:
                self.save()  # 保存模型
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        """执行单步训练"""
        batch = th.cat((batch, cond), dim=1)
        cond = {}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()  # 更新EMA参数
        self._anneal_lr()  # 调整学习率
        self.log_step()  # 记录步数
        return sample

    def forward_backward(self, batch, cond):
        """前向和反向传播"""
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]
            loss = (losses["loss"] * weights).mean()

            # 保存损失到CSV文件
            if self.step % self.log_interval == 0:
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.step + self.resume_step, loss.item()])  # 保存步数和损失

            self.mp_trainer.backward(loss)
            return sample

    def _update_ema(self):
        """更新EMA参数"""
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        """退火学习率"""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """记录步数和样本数"""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """保存模型和优化器状态"""
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"保存模型 {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

def parse_resume_step_from_filename(filename):
    """从文件名解析恢复步数"""
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    """获取日志目录"""
    return logger.get_dir()

def find_resume_checkpoint():
    """查找恢复检查点"""
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    """查找EMA检查点"""
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    """记录损失字典"""
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)