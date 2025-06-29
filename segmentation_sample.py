import argparse
import os
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# 设置随机种子以确保结果可重复
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    # 将图像归一化到0-1范围
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def save_image(img, filename, caption):
    # 保存归一化后的图像到本地文件
    plt.figure()
    plt.imshow(img.cpu().numpy(), cmap='gray')
    plt.title(caption)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def dice_score(pred, targs):
    # 计算Dice系数，用于评估分割效果
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


def main():
    # 解析命令行参数
    args = create_argparser().parse_args()
    # 初始化分布式训练环境
    dist_util.setup_dist()
    # 配置日志
    logger.configure()

    # 创建输出目录（如果不存在）
    os.makedirs('./results', exist_ok=True)

    # 创建模型和扩散过程
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 加载BRATS测试数据集
    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    # 加载预训练模型权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # 将模型移动到指定设备（GPU/CPU）
    model.to(dist_util.dev())
    if args.use_fp16:
        # 转换为半精度浮点数以加速计算
        model.convert_to_fp16()
    # 设置模型为评估模式
    model.eval()

    # 循环生成样本直到达到指定数量
    while len(all_images) * args.batch_size < args.num_samples:
        # 从数据加载器获取一批数据
        b, path = next(data)  # should return an image from the dataloader "data"
        # 生成与输入相同形状的随机噪声通道
        c = th.randn_like(b[:, :1, ...])
        # 将输入图像和噪声通道拼接
        img = th.cat((b, c), dim=1)
        # 提取切片ID
        slice_ID = path[0].split("/", -1)[3]

        # 保存输入图像
        save_image(visualize(img[0, 0, ...]), f'./results/{slice_ID}_input0.png', "img input0")
        save_image(visualize(img[0, 1, ...]), f'./results/{slice_ID}_input1.png', "img input1")
        save_image(visualize(img[0, 2, ...]), f'./results/{slice_ID}_input2.png', "img input2")
        save_image(visualize(img[0, 3, ...]), f'./results/{slice_ID}_input3.png', "img input3")
        save_image(visualize(img[0, 4, ...]), f'./results/{slice_ID}_input4.png', "img input4")

        # 开始采样过程
        logger.log("sampling...")

        # 记录采样时间
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        # 生成指定数量的样本集合
        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            # 选择采样函数（基于DDIM或普通扩散）
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            # 执行采样过程
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            # 打印单次采样的时间
            print('time for 1 sample', start.elapsed_time(end))

            # 保存生成的样本
            s = th.tensor(sample)
            save_image(visualize(sample[0, 0, ...]), f'./results/{slice_ID}_output{i}.png', "sampled output")
            # 保存生成的掩码张量
            th.save(s, f'./results/{slice_ID}_output{i}.pt')


def create_argparser():
    # 定义默认参数
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5  # 样本集合的数量
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()