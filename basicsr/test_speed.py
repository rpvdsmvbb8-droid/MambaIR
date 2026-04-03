import logging
import torch
import os
from os import path as osp
import sys
import time

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

    # create model
    model = build_model(opt)
    model.net_g.eval()

    # generate random input tensor
    scale = opt['scale']
    img_size = 256
    dummy_input = torch.randn(1, 3, img_size, img_size).cuda()

    # warm up
    logger.info("Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model.net_g(dummy_input)
    torch.cuda.synchronize()

    # benchmark
    logger.info("Starting Benchmark...")
    repeats = 100
    timings = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(repeats):
            starter.record()
            _ = model.net_g(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    avg_time = sum(timings) / repeats
    fps = 1000 / avg_time

    logger.info('-------------------------------------------')
    logger.info(f'Speed Test Result (Input: {img_size}x{img_size})')
    logger.info(f'Total Time: {sum(timings) / 1000:.4f} s')
    logger.info(f'Average Time: {avg_time / 1000:.4f} s')
    logger.info(f'FPS: {fps:.2f}')
    logger.info('-------------------------------------------')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
