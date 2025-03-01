import torch
from unet import Unet
from train import DDPM, sample


if __name__ == '__main__':
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型结构
    denoise_model = Unet().to(device)
    ddpm = DDPM(
        unet=denoise_model,
        t=1000,
        device=device
    )
    # 加载模型参数
    ddpm.load_state_dict(torch.load('50-epochs-log/ddpm-model-50.ckpt'))
    # 设置保存路径
    sample(ddpm, 1000, device, '50-result/', 16)
