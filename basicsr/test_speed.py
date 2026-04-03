import torch
import time
import logging
import os
import sys
from os import path as osp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from basicsr.utils import get_root_logger, get_time_str
from basicsr.utils.options import dict2str, parse_options

try:
    from archs.mambairv2_arch import MambaIRv2Light
except ImportError:
    from basicsr.archs.mambairv2_arch import MambaIRv2Light


def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # 1. 直接实例化网络结构 (根据 MambaIRv2 的定义)
    # 注意：这里的参数需要和你训练时的 yaml 配置一致
    # 从你的截图看，这是一个 x2 的轻量级模型
    model = MambaIRv2Light(
        inp_channels=3,
        out_channels=3,
        dim=48,            # 对应截图里的 Conv2d(48, 48...)
        num_blocks=[4, 4], # 这是一个常见的配置，如果报错请检查你的 yaml
        upscale=2,         # 对应 x2 模型
        drop_path_rate=0.0
    )

    # 2. 加载权重
    load_path = opt['path'].get('pretrain_network_g', None)
    if load_path:
        logger.info(f"Loading model from [{load_path}] ...")
        # 这里处理权重字典，通常只需要 'params' 键
        load_net = torch.load(load_path, map_location=torch.device('cuda'))
        if 'params' in load_net:
            load_net = load_net['params']
        model.load_state_dict(load_net, strict=True)

    model = model.cuda()
    model.eval()

    # 3. 预热
    logger.info("Warming up GPU...")
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    # 4. 正式测试
    logger.info("Starting Benchmark...")
    timings = []
    with torch.no_grad():
        for _ in range(100):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    avg_time = sum(timings) / len(timings) / 1000.0  # 转换为秒
    fps = 1.0 / avg_time

    logger.info("--------------------------------------------------")
    logger.info(f"Speed Test Result (Input: 256x256)")
    logger.info(f"Average Time: {avg_time:.4f} s")
    logger.info(f"FPS: {fps:.2f}")
    logger.info("--------------------------------------------------")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
