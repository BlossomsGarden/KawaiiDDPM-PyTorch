import torch
from Unet import Unet
from train import DDPM, sample


if __name__ == '__main__':
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型结构
    denoise_model = Unet().to(device)
    ddpm = DDPM(
        denoise_model=denoise_model,
        t=1000,
        device=device
    )
    # 加载模型参数
    ddpm.load_state_dict(torch.load('ddpm_model.ckpt'))
    sample(ddpm, 1000, device, 16)
