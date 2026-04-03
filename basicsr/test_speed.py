import logging
import torch
import os
import time  # 新增：用于计时
from os import path as osp
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    
    model = build_model(opt)
    model.net_g.eval() # 切换为评估模式
    net_g = model.net_g.cuda()

    dummy_input = torch.randn(1, 3, 256, 256).cuda() 
    
    logger.info("Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = net_g(dummy_input)
    

    logger.info("Starting Benchmark...")
    torch.cuda.synchronize() # 确保之前的代码执行完毕
    start_time = time.time()

    with torch.no_grad():
        for _ in range(200): # 运行 200 次取平均
            _ = net_g(dummy_input)
    
    torch.cuda.synchronize() # 确保所有计算完成
    end_time = time.time()

    # 5. 计算结果
    total_time = end_time - start_time
    avg_time = total_time / 200
    fps = 1.0 / avg_time

    logger.info("-" * 40)
    logger.info(f"Speed Test Result (Input: 256x256)")
    logger.info(f"Total Time: {total_time:.4f} s")
    logger.info(f"Average Time: {avg_time:.4f} s")
    logger.info(f"FPS: {fps:.2f}")
    logger.info("-" * 40)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
