# ----------------------------------------------------------------------
# 1. 数据集预处理
# ----------------------------------------------------------------------
from torchvision import transforms
# transforms.RandomRotation(15),
# transforms.RandomCrop(64, padding=4),
# BYD 你分类用这3个就算了，生成也加？
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(64),
    transforms.ToTensor()
])


#  ----------------------------------------------------------------------
# 2. 数据集加载
#  ----------------------------------------------------------------------
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
# 重定义PyTorch的Dataset类
class MyAnimeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)]

    # PyTorch 的 Dataset 类中__len__() 返回数据集的样本数量，通常通过 len(ds) 来调用
    def __len__(self):
        return len(self.images)

    # __getitem__ 用于实现索引访问的特殊方法。隐式，使用索引访问对象如 obj[index]，Python 会自动调用 __getitem__
    # 请注意，经handle2dataset处理后，文件夹内图片从1.png开始
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, f"{idx + 1}.png")
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# 创建自己的数据集实例
train_dataset = MyAnimeDataset(
    dataset_dir="./AnimeFaces64",
    transform=transform_train
)
# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)


# ----------------------------------------------------------------------
# 3. DDPM模型
# ----------------------------------------------------------------------
import torch.nn as nn
import torch
from unet import Unet
# DenoiseDiffusion
class DDPM(nn.Module):
    # n_steps即训练时扩散的步长
    def __init__(self, unet: Unet, t: int, device: torch.device):
        super(DDPM, self).__init__()
        # 创建一个从 0.0001 到 0.02 等间隔的数组，长度即步长，表示扩散过程中的 β 值。
        # 同时计算衰减系数计算 α 与 α_bar
        self.beta = torch.linspace(0.0001, 0.02, t).to(device)
        self.sigma2 = self.beta # 本论文给出了证明，Sampling中 σt^2 = βt，
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.unet = unet
        self.device=device
        self.t = t

    # 计算加噪步长t后的结果xt图像，即 p_sample
    def diffuse(self, x0: torch.Tensor, t:torch.Tensor, noise=None):
        # 若未传入噪声，则生成一个与x0形状相同的标准正态分布噪声（x0只贡献形状而非）
        if noise is None:
            noise = torch.randn_like(x0)
        # 直接取[]得到的alpha_bar_t是一维张量（如[batch_size]），需要扩展为[batch_size, 1, 1, 1]
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        # xt = √α_bar_t * x0 + √(1 - α_bar_t) * ε
        # 其中 ε ~ N(0, I)，那么 xt ~ N(√α_bar_t * x0, (1 - α_bar_t) * I)
        mean = (alpha_bar_t ** 0.5) *x0
        variance = 1 - alpha_bar_t
        return mean + (variance ** 0.5) * noise

    # Algorithm1的具体实现，即 q_sample
    def MSEloss(self, x0: torch.Tensor, noise=None):
        batch_size = x0.shape[0]
        # 抽样 t
        t = torch.randint(0, self.t, (batch_size,), dtype=torch.long, device=self.device)
        # 未传入则从 N(0, I) 中随机抽取
        if noise is None:
            noise = torch.randn_like(x0)
        # 加噪过程计算 xt
        xt = self.diffuse(x0, t, noise)
        # 预测 xt-1 -> xt 的噪声
        predict = self.unet(xt, t)
        return  nn.functional.mse_loss(noise, predict)

    # 即计算 p(xt-1 | xt)
    # xt-1公式在Algorithm2中，其中可证明σt2=βt，因此噪声前的系数为√βt
    def denoise(self, xt: torch.Tensor, t: torch.Tensor):
        # 直接取[]得到的是一维张量（如[batch_size]），需要扩展为[batch_size, 1, 1, 1]
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        sigma2_t = self.sigma2[t].reshape(-1, 1, 1, 1)

        predict_noise = self.unet(xt, t)
        random_noise = torch.randn(xt.shape, device=self.device)
        mean = 1 / (alpha_t ** 0.5) * (xt - ((1 - alpha_t) / (1 - alpha_bar_t) ** 0.5) * predict_noise)
        variance = sigma2_t
        return mean + (variance ** 0.5) * random_noise


# ----------------------------------------------------------------------
# 4. Training 和 Sampling
# ----------------------------------------------------------------------
# Algorithm1
# 共 78698 张图片，RTX4090/30G 一个 epoch 约 4 分钟
def train(ddpm: DDPM, epoch: int, optimizer: torch.optim.Optimizer):
    running_loss = 0.0
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)):
        data = data.to(device)  # 转换数据格式
        # PyTorch 中，梯度是通过反向传播计算的，并且默认是累加的（每次计算梯度时会加到现有的梯度上）
        # 因此每个batch开始时需要清零梯度，避免梯度不断累积导致不正确的参数更新
        optimizer.zero_grad()
        # 通过预测结果与目标，计算损失函数
        loss = ddpm.MSEloss(data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  # running_loss为当前epoch总损失，.item()将数据类型由tensor转为python数据

    train_loss = running_loss / len(train_loader)
    print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}")

# Algorithm2
# 供训练完毕后期调用，批量去噪生成新图，默认一次生成16张、每张图片3通道、图片大小64*64
from torchvision.utils import make_grid
def sample(ddpm_model: DDPM, t: int, device: torch.device, save_path: str, batch_size: int = 16):
    ddpm_model.eval()  # 设置为评估模式
    newDir(save_path)   # 搞好保存路径
    with torch.no_grad():
        x = torch.randn([batch_size, 3, 64, 64], device=device)
        # 每张图片对应一个列表，存储其降噪过程中的图像
        denoising_process = [[] for _ in range(batch_size)]
        prompt = f"{batch_size} images, {t} steps. Denoising"
        for _t in tqdm(range(t), desc=prompt, ncols=100):
            # 计算当前时间步（从 t-1 到 0）
            current_step = t - _t - 1
            # 将当前时间步转换为张量
            tensor_t = torch.tensor([current_step], dtype=torch.long, device=device)
            # 降噪一步
            x = ddpm_model.denoise(x, tensor_t)
            # 每 19 步保存一次降噪结果（总共保存 20 次）
            if current_step % (t // 20) == 0 or current_step == 0:
                # 将每张图片的当前状态保存到对应的列表中
                for i in range(batch_size):
                    denoising_process[i].append(x[i].clamp(0, 1))

        # 降噪完毕后将每张图片的降噪过程拼接为一行 20 列的图片，并保存为 1.png 到 (batch_size).png
        for i in range(batch_size):
            # 将降噪过程中的所有图片拼接为一行
            process_grid = make_grid(denoising_process[i], nrow=20, padding=2, normalize=True)
            # 将张量转换为 PIL 图像
            to_pil = transforms.ToPILImage()
            process_image = to_pil(process_grid)
            process_image.save(os.path.join(save_path, f"{i + 1}.png"))

        # 将降噪完毕的结果按4*4保存一下
        grid = make_grid(x.clamp(0, 1), nrow=4, padding=2, normalize=True)
        to_pil = transforms.ToPILImage()
        grid_image = to_pil(grid)
        grid_image.save(os.path.join(save_path, str(batch_size) + "-imgs-final.png"))

# ----------------------------------------------------------------------
# 5. 设置参数开训
# ----------------------------------------------------------------------
from tqdm import tqdm
from utils import newDir

if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-5
    epochs = 50
    steps = 1000

    # 你不加to(device)输入全是cpu，模型权重全是cuda:0，你接着顽咯
    denoise_model = Unet().to(device)
    ddpm = DDPM(
        unet=denoise_model,
        t=steps,
        device=device
    )
    # 创建优化器
    optimizer = torch.optim.Adam(
        denoise_model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):
        train(ddpm=ddpm, epoch=epoch, optimizer=optimizer)
        # 每10轮看一下当前训练成果并保存一次模型
        if epoch % 10 == 9 :
            save_path = str(epoch) + "-epochs-log"
            sample(ddpm, steps, device, save_path, 16)
            torch.save(ddpm.state_dict(), os.path.join(save_path, 'ddpm-model-'+ str(epoch) + '.ckpt'))

