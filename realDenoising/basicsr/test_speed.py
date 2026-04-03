import logging
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse

try:
    from basicsr.train import parse_options
except ImportError:
    from basicsr.utils.options import parse_options


def main():
    opt = parse_options(is_train=False)

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    model = create_model(opt)
    model.eval() 
    
    net_g = model.net_g
    net_g.cuda()

    speed_opt = opt.get('speed_test', {})
    
    if not speed_opt:
        logger.error("配置文件中缺少 'speed_test' 字段，无法进行速度测试。")
        return

    use_random_input = speed_opt.get('use_random_input', True)
    input_size = speed_opt.get('input_size', [256, 256])
    warmup = speed_opt.get('warmup', 50)
    repeats = speed_opt.get('repeats', 200)

    logger.info(f"开始速度测试: 输入尺寸={input_size}, 预热={warmup}, 重复={repeats}")

    if use_random_input:
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).cuda()
        logger.info("使用随机张量作为输入。")
    else:
        test_loaders = []
        for phase, dataset_opt in sorted(opt['datasets'].items()):
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt.get('dist', False), sampler=None, seed=opt['manual_seed'])
            test_loaders.append(test_loader)
        
        val_data = next(iter(test_loaders[0]))
        dummy_input = val_data['lq'].cuda()
        logger.info(f"使用数据集 {opt['datasets']['test_1']['name']} 的真实图片作为输入。")

    logger.info(f"预热中 (Warm-up {warmup} 次)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = net_g(dummy_input)
    torch.cuda.synchronize()

    logger.info(f"正式测试中 (Repeats {repeats} 次)...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(repeats):
            starter.record()
            _ = net_g(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 必须同步，否则测的是异步启动时间
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    avg_time = sum(timings) / repeats
    fps = 1000 / avg_time
    
    logger.info(f"--------------------------------------------------")
    logger.info(f"平均延迟 (Latency): {avg_time:.4f} ms")
    logger.info(f"吞吐量 (FPS): {fps:.2f}")
    logger.info(f"显存峰值占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    logger.info(f"--------------------------------------------------")

if __name__ == '__main__':
    main()
