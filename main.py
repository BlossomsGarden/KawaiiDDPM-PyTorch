# ----------------------------------------------------------------------
# 1. 数据处理流程
# ----------------------------------------------------------------------
from torchvision import transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(60, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])


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
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, f"{idx}.png")
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
# Residual
# 残差块通过引入残差魅力时刻，避免深层网络中的梯度消失问题，提升了网络的训练效率和稳定性
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super(ResidualBlock, self).__init__()

        # 步长编码器
        # 处理步长T的向量维度，使之与Block1输出的图片数据结果相同，方便融合
        self.time_embedding=nn.Sequential(
            nn.Linear(time_channels, out_channels),
            lambda x: x * torch.sigmoid(x),
            lambda x: x[:, :, None, None]
        )

        # Residual 模块：两块block做特征提取、一块残差处理器将block的结果和原输入相加
        # 第一块对图片数据处理使in_channels = out_channels
        self.block1 = nn.Sequential(
            # 对输入特征图进行分组归一化，提升训练稳定性
            # batch_size较大时应首选BatchNorm，但它在每个Batch计算均值和方差，若Batch_size过小，结果的均值和方差可能不稳定，导致模型训练过程不稳定，效果差

            # 生成式模型最好使用小批次任务的原因：
            # 往往需要进行复杂的采样过程，而每个样本生成的过程相对较长，导致批次处理的数量可能较少
            # 每次生成图像或其他高维数据时，可能会消耗大量内存，因此在训练时小batch_size降低内存压力
            # 往往关注样本的质量和多样性，因此小批次训练可以避免对大批次数据进行过多的平均化，使得模型能够更好地捕捉数据的细节
            nn.GroupNorm(32, in_channels),
            lambda x: x * torch.sigmoid(x),     # 即Swish()，一种平滑的非线性激活函数，增强非线性表达能力
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        )

        #第二块对图片步长形成的复合数据处理
        self.block2=nn.Sequential(
            nn.GroupNorm(32, in_channels),
            lambda x: x * torch.sigmoid(x),  # 即Swish()，一种平滑的非线性激活函数，增强非线性表达能力
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        )

        # 残差处理器
        # in_channels = out_channels时，做恒等映射(Identity)，即啥也不干，可以不做任何处理直接相加；
        # in_channels != out_channels时，需要对输入数据做一次卷积，将其通道数变成一致再相加
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))


    # 模型有2个输入：
    # x：步长T下的图片数据 [(batch_size, in_channels, height, width)]
    # t：步长T [(batch_size, time_channel)]
    def forward(self, x, t):
        x2 = self.block1(x)
        # 处理步长T形成的time_embedding向量，通过线性层使time_c变为out_c，再和输入数据的特征图相加
        x2 += self.time_embedding
        x2 = self.block2(x2)
        # 残差魅力时刻
        return x2 + self.shortcut(x)



# TimeEmbedding
# 将步长T这个整数包装成维度为time_channel的向量 [(batch_size, time_channel)]
# （这个包装方式和Transformer中函数式位置编码的包装方式一致）
# （之后只需要再通过一个简单的线性层，将其维度从time_channel转变为对应特征图的out_channel，就能够和特征图相加）
import math
class TimeEmbedding(nn.Module):
    def __init__(self, time_channels: int):
        super().__init__()
        self.time_channels = time_channels

        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_channels // 4, self.time_channels),  # python中 / 返回浮点数，//才舍去小数
            lambda x: x * torch.sigmoid(x),     # 即Swish()，一种平滑的非线性激活函数，增强非线性表达能力
            nn.Linear(self.time_channels, self.time_channels),
        )

    def forward(self, t: torch.Tensor):
        # 以下转换方法和Transformer的位置编码一致
        half_dim = self.time_channels // 8
        # C_embedding 是一个用于确定时间嵌入的缩放因子，它帮助计算不同频率的嵌入向量。
        # 这个缩放因子是为生成时间步的频率信息（sin 和 cos）所需要的，控制着生成正弦和余弦频率时的细节。
        # 该公式的设计基于 Transformer 中位置编码的理念。
        # emb 控制了频率（即周期），使得时间编码能够捕捉到足够多的时间步信息，从而允许模型处理长时间序列和周期性的信息。
        C_embedding = math.log(10_000) / (half_dim - 1)
        C_embedding = torch.exp(torch.arange(half_dim, device=t.device) * -torch.tensor(C_embedding, device=t.device))
        # 将时间步长 t 与 C_embbing 进行广播操作（broadcasting）
        # t[:, None] 会将 t 变成列向量，而 C_embbing[None, :] 将 C_embbing 转换为行向量。这样就能将每个时间步的 t 和每个频率值进行逐元素相乘，生成时间步的编码。
        C_embedding = t[:, None] * C_embedding[None, :]
        # 计算 emb 的正弦和余弦值，并将它们拼接（cat）在一起，形成一个包含正弦和余弦的时间编码
        time_embedding = torch.cat((C_embedding.sin(), C_embedding.cos()), dim=1)
        time_embedding=self.time_embedding(time_embedding)
        return time_embedding



