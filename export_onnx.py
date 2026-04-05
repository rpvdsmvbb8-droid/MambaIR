import torch
import torch.onnx
from basicsr.archs.mambairv2light_arch import MambaIRv2Light

def main():
    # --- 1. 根据你的 YAML 配置定义参数 ---
    # 这些参数必须和你加载的 .pth 模型训练时的参数完全一致
    model_params = dict(
        img_size=64,          # YAML 中的设置
        patch_size=1,         # YAML 中的设置
        in_chans=3,           # YAML 中的设置 (输入是 3 通道 RGB)
        embed_dim=48,         # YAML 中的设置
        d_state=8,            # YAML 中的设置
        depths=[5, 5, 5, 5],  # YAML 中的设置
        num_heads=[4, 4, 4, 4], # YAML 中的设置
        window_size=16,       # YAML 中的设置
        inner_rank=32,        # YAML 中的设置
        num_tokens=64,        # YAML 中的设置
        convffn_kernel_size=5, # YAML 中的设置
        mlp_ratio=1.0,        # YAML 中的设置
        upscale=2,            # YAML 中的设置
        img_range=1.0,        # YAML 中的设置
        upsampler='pixelshuffledirect', # YAML 中的设置
        resi_connection='1conv' # 代码中的默认值，通常不用改
    )

    # --- 2. 实例化模型 ---
    # 注意：这里实例化的是纯 PyTorch 模型，不包含 BasicSR 的 Trainer 包装
    model = MambaIRv2Light(**model_params)
    
    # --- 3. 加载预训练权重 ---
    # 确保路径正确
    checkpoint_path = "experiments/pretrained_models/mambairv2_lightSR_x2.pth" 
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 根据你的 YAML，权重在 'params' 键下
    param_key = 'params' 
    model.load_state_dict(checkpoint[param_key], strict=True)
    model.eval() # 切换到推理模式

    # --- 4. 构造虚拟输入 ---
    # Batch=1, Channel=3, Height=64, Width=64
    # 注意：虽然模型可以处理任意尺寸，但 ONNX 导出通常需要固定一个典型的输入尺寸
    dummy_input = torch.randn(1, 3, 64, 64)

    # --- 5. 导出 ONNX ---
    output_path = "MambaIRv2Light_x2.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,  # 存储训练参数
        opset_version=13,    # 推荐使用 13 以支持更多算子
        do_constant_folding=True,  # 优化常量
        input_names=['input'],     # 输入名
        output_names=['output'],   # 输出名
        dynamic_axes={             # 允许动态 batch 和分辨率 (可选，但推荐)
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'out_height', 3: 'out_width'}
        }
    )
    
    print(f" 导出成功！文件已保存为 {output_path}")

if __name__ == "__main__":
    main()
