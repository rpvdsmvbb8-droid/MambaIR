import time
import torch
import yaml
import importlib

def main():
    config_path = 'options/test_speed/test_MambaIRv2_lightSR_x2.yml'

    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)

    network_opt = opt['network_g']
    model_type = network_opt.pop('type')

    try:
        module = importlib.import_module('mambair.archs.mambairv2_arch')
    except ImportError:
        module = importlib.import_module('realDenoising.basicsr.archs.mambairv2_arch')

    model_class = getattr(module, model_type)
    model = model_class(**network_opt).cuda()

    # 配置测试参数
    repeats = opt['speed_test']['repeats']
    input_size = opt['speed_test']['input_size']
    img_range = opt['network_g']['img_range']

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).cuda()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 测速
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / repeats
    print(f"Input Size: {input_size}x{input_size}")
    print(f"Average Inference Time: {avg_time:.4f} s")
    print(f"FPS: {1/avg_time:.2f}")

if __name__ == '__main__':
    main()
