import logging
import time
import torch
from os import path as osp

from basicsr.models import create_model
from basicsr.utils import get_root_logger, get_env_info, dict2str
from basicsr.utils.options import parse_options

def main():
    # 1. 解析配置
    opt = parse_options(is_train=False)
    logger = get_root_logger(logger_name='basicsr')
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # 2. 获取速度测试参数
    speed_opt = opt.get('speed_test', {})
    warmup = speed_opt.get('warmup', 50)
    repeats = speed_opt.get('repeats', 200)
    input_size = speed_opt.get('input_size', [256, 256]) # 读取配置中的分辨率

    # 3. 创建模型
    model = create_model(opt)
    model.net_g.eval()

    # 4. 准备随机输入
    # 使用配置中的尺寸生成随机张量 (Batch, Channel, Height, Width)
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).cuda()

    # 5. 预热
    logger.info(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.net_g(input_tensor)
    torch.cuda.synchronize()

    # 6. 正式测速
    logger.info(f"Running benchmark for {repeats} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model.net_g(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()

    # 7. 计算结果
    total_time = end_time - start_time
    avg_time = total_time / repeats
    fps = 1.0 / avg_time

    logger.info("-" * 40)
    logger.info(f"Benchmark Results (Input: {input_size[0]}x{input_size[1]})")
    logger.info(f"Total Time: {total_time:.4f} s")
    logger.info(f"Average Inference Time: {avg_time:.4f} s")
    logger.info(f"Frames Per Second (FPS): {fps:.2f}")
    logger.info("-" * 40)

if __name__ == '__main__':
    main()
