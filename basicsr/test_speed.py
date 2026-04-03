import logging
import torch
import os
from os import path as osp
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    model = build_model(opt)
    model.setup()
    model.net_g.eval()

    if hasattr(model.net_g, 'module'):
        net = model.net_g.module
    else:
        net = model.net_g

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    input_size = opt.get('speed_test', {}).get('input_size', 64)
    warmup = opt.get('speed_test', {}).get('warmup', 50)
    repeats = opt.get('speed_test', {}).get('repeats', 100)

    logger.info(f"Input Size: {input_size}x{input_size}")
    logger.info(f"Warmup: {warmup} iterations")
    logger.info(f"Benchmark: {repeats} iterations")

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = net(dummy_input)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(repeats):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    avg_time = sum(timings) / repeats
    fps = 1000 / avg_time

    logger.info("--------------------------------------------------")
    logger.info(f"Speed Test Result (Input: {input_size}x{input_size})")
    logger.info(f"Total Time: {sum(timings) / 1000:.4f} s")
    logger.info(f"Average Time: {avg_time / 1000:.4f} s")
    logger.info(f"FPS: {fps:.2f}")
    logger.info("--------------------------------------------------")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
