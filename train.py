# ----------------------------------------------------------------------
# 1. 数据集预处理
# ----------------------------------------------------------------------
import os
from PIL import Image
from torchvision import transforms
# transforms.RandomHorizontalFlip(),
# transforms.RandomRotation(15),
# transforms.RandomCrop(64, padding=4),
# BYD 你分类用这3个就算了，生成也加？
transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

# 将AnimeFaceDataset内的图片全部转为png格式，再移动到AnimeFace64中，并且以顺序命名
def handle2dataset():
    # 定义文件夹路径
    anime_face_dataset_dir = "AnimeFaceDataset"  # 原始图片文件夹
    anime_faces64_dir = "AnimeFaces64"  # 目标图片文件夹
    # 获取AnimeFaces64文件夹中已有的文件数
    # no = len(os.listdir(anime_faces64_dir))
    no=21551
    # 遍历AnimeFaceDataset文件夹中的所有jpg文件
    for filename in tqdm(os.listdir(anime_face_dataset_dir), desc="Denoising", ncols=100):
        if filename.endswith(".jpg"):
            # 构建原始文件路径
            jpg_path = os.path.join(anime_face_dataset_dir, filename)
            with Image.open(jpg_path) as img:
                # 图片的宽度和高度小于64的直接跳过
                width, height = img.size
                if width < 64 or height < 64:
                    continue

                # 生成新的文件名
                no += 1
                new_filename = f"{no}.png"
                new_path = os.path.join(anime_faces64_dir, new_filename)
                # 转换为png格式并保存
                img.save(new_path, "PNG")

    print("图片转换和重命名完成！")

#  ----------------------------------------------------------------------
# 2. Dataset和DataLoader
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
from Unet import Unet


# DenoiseDiffusion
class DDPM(nn.Module):
    # t即扩散的步长
    def __init__(self, denoise_model: Unet, t: int, device: torch.device):
        super(DDPM, self).__init__()
        # 创建一个从 0.0001 到 0.02 等间隔的数组，长度即步长，表示扩散过程中的 β 值。
        # 同时计算衰减系数计算 α 与 α_bar
        self.beta = torch.linspace(0.0001, 0.02, t).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # 即传入的基于Algorithm1训练出来的去噪模型
        self.denoise_model = denoise_model
        self.device=device
        self.t = t

    # 计算加噪步长t后的结果xt图像
    def diffuse(self, x0: torch.Tensor, t:torch.Tensor, noise=None):
        # 若未传入噪声，则生成一个与x0形状相同的标准正态分布噪声（x0只贡献形状而非）
        if noise is None:
            noise = torch.randn_like(x0).to(self.device)

        # 直接取[]得到的alpha_bar_t是一维张量（如[batch_size]），需要扩展为[batch_size, 1, 1, 1]
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)

        # xt = √α_bar_t * x0 + √(1 - α_bar_t) * ε
        # 其中 ε ~ N(0, I)，那么 xt ~ N(√α_bar_t * x0, (1 - α_bar_t) * I)
        mean = (alpha_bar_t ** 0.5) *x0
        variance = 1 - alpha_bar_t
        return mean + (variance ** 0.5) * noise

    # Algorithm1的具体实现
    def MSEloss(self, x0: torch.Tensor, noise=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.t, (batch_size,), dtype=torch.long, device=self.device)
        if noise is None:
            noise = torch.randn_like(x0).to(self.device)
        xt = self.diffuse(x0, t, noise)
        predict = self.denoise_model(xt, t)
        loss = nn.functional.mse_loss(noise, predict)
        return  loss

    # 即计算 p(xt-1 | xt)
    # xt-1公式在Algorithm2中，其中可证明σt2=βt，因此噪声前的系数为√βt
    def denoise(self, xt: torch.Tensor, t: torch.Tensor):
        noise = torch.randn(xt.shape, device=self.device)
        # 直接取[]得到的是一维张量（如[batch_size]），需要扩展为[batch_size, 1, 1, 1]
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)
        return 1/(alpha_t ** 0.5) * (xt - (1 - alpha_t)/((1 - alpha_bar_t) ** 0.5) * self.denoise_model(xt, t)) + (beta_t ** 0.5) * noise


# ----------------------------------------------------------------------
# 4. 训练与验证函数
# ----------------------------------------------------------------------
# Algorithm1
from tqdm import tqdm
def train(epoch, device):
    running_loss = 0.0

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)):
        data = data.to(device)  # 转换数据格式
        # PyTorch 中，梯度是通过反向传播计算的，并且默认是累加的（每次计算梯度时会加到现有的梯度上）
        # 每次梯度更新前需要清零梯度，避免梯度不断累积导致不正确的参数更新
        optimizer.zero_grad()
        # 通过预测结果与目标，计算损失函数
        loss = ddpm.MSEloss(data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # running_loss为当前epoch总损失，.item()将数据类型由tensor转为python数据

    return running_loss / len(train_loader)

# Algorithm2
# 供训练完毕后期调用，批量去噪生成新图，默认一次生成16张、每张图片3通道、图片大小64*64
from torchvision.utils import make_grid
def sample(ddpm_model: DDPM, t: int, device: torch.device, save_path: str, batch_size: int = 16):
    ddpm_model.eval()  # 设置为评估模式
    with torch.no_grad():
        x = torch.randn([batch_size, 3, 64, 64], device=device)
        # 降噪n次，使用tqdm显示进度条
        for _t in tqdm(range(t), desc="Denoising", ncols=100):
            # 传入的t的格式是torch.Tensor，参见DDPM.MSELoss()中t的生成
            tensor_t = torch.tensor([_t], dtype=torch.long, device=device)
            x =ddpm_model.denoise(x, tensor_t)

            # 每40次降噪保存一次看看效果
            if _t != 0 and _t %40 ==0:
                # 将16张图片拼接为4x4的网格
                grid = make_grid(x.clamp(0, 1), nrow=4, padding=2, normalize=True)
                # 将张量转换为PIL图像
                to_pil = transforms.ToPILImage()
                grid_image = to_pil(grid)
                grid_image.save(os.path.join(save_path, "sample_eval_grid_" + str(_t) + ".png"))



# ----------------------------------------------------------------------
# 5. 开训
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 2e-5
    epochs = 100
    steps = 1000

    # 你不加to(device)输入全是cpu，模型权重全是cuda:0，你接着顽咯
    denoise_model = Unet().to(device)
    ddpm = DDPM(
        denoise_model=denoise_model,
        t=steps,
        device=device
    )
    # 创建优化器
    optimizer = torch.optim.Adam(
        denoise_model.parameters(),
        lr=learning_rate
    )
    # 共78767张图片，RTX4090/30G 一个epoch约6分钟
    for epoch in range(epochs):
        train_loss = train(epoch, device)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f}")
        # 每20轮看一下当前训练成果并保存一次模型
        if epoch != 0 and epoch % 20 == 0 :
            save_path = str(epoch)+"-epochs-log"
            os.makedirs(save_path)
            sample(ddpm, steps, device, save_path, 16)
            torch.save(ddpm.state_dict(), os.path.join(save_path, 'ddpm_model'+ str(epoch) + '.ckpt'))

