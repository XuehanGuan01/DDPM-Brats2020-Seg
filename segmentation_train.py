"""
训练图像分割的扩散模型。
"""
import sys
import argparse
import csv
import os
import torch as th
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()

    # 设置多GPU训练环境
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    # 验证数据目录
    if not os.path.exists(args.data_dir):
        logger.log(f"错误：数据目录 {args.data_dir} 不存在！")
        sys.exit(1)

    logger.log("创建数据加载器...")
    try:
        ds = BRATSDataset(args.data_dir, test_flag=False)
        if len(ds) == 0:
            logger.log(f"错误：数据集为空！请检查 {args.data_dir} 中的文件结构和命名。")
            logger.log("建议：确保每个样本目录包含所有模态文件（t1, t1ce, t2, flair, seg）。")
            sys.exit(1)
        
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # 禁用多进程
            pin_memory=True,
            drop_last=True
        )
        data = iter(datal)
        logger.log(f"成功加载数据集，样本数：{len(ds)}")
    except Exception as e:
        logger.log(f"数据加载失败：{e}")
        sys.exit(1)

    logger.log("创建模型和扩散过程...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    # 初始化CSV文件
    metrics_file = os.path.join(logger.get_dir(), "training_metrics.csv")
    try:
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Loss', 'Dice'])
        logger.log(f"创建CSV文件：{metrics_file}")
    except IOError as e:
        logger.log(f"无法创建CSV文件：{e}")

    logger.log("开始训练...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        metrics_file=metrics_file
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="./data/training",
        log_dir="./logs",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_workers=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()