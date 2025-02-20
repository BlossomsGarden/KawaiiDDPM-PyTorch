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
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
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


# Attention
# 与Transformer中的multi-head attention原理及实现方式一致（多头机制使模型可以并行关注不同模式的信息（如形状、颜色、纹理）增强表达能力）
#
# 作用：
# 1、捕捉全局依赖：传统卷积层受限于局部感受野，而注意力机制可以建模图像中任意两个像素的关系，帮助模型理解全局结构（如对称性、物体间关系）
# 2、增强去噪能力：在扩散模型的去噪过程中，某些区域的噪声可能与其他区域相关（如边缘、纹理连续性）。注意力机制通过加权聚合信息，帮助模型更准确地预测噪声。
# 3、动态调整权重：不同时间步（扩散阶段）可能需要不同的关注模式。注意力机制允许模型自适应地调整权重，适应不同噪声水平下的生成需求。
class AttentionBlock(nn.Module):
    # channels：输入特征图的通道数
    # n_heads：注意力头的数量（默认为1）
    # d_k：每个注意力头的向量维度（默认为n_channels）。
    def __init__(self, channels, n_heads=1, d_k=None):
        super(AttentionBlock, self).__init__()

        if d_k is None:
            d_k = channels
        self.norm = nn.GroupNorm(32, channels)
        # 将输入特征映射到Q、K、V矩阵的组合，输出维度为n_heads * d_k * 3。
        self.projection = nn.Linear(channels, n_heads * d_k * 3)
        # 将注意力模块的结果合并回原始通道
        self.output = nn.Linear(n_heads * d_k, channels)
        # 用于缩放点积注意力得分，防止数值过大。
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    # 可以忽略步长T
    def forward(self, x):
        # 将输入特征图从 (batch_size, channels, height, width) 转换为 (batch_size, height*width, channels)，使其适应序列处理形式
        batch_size, n_channels, height, width = x.shape
        # 通过投影层生成Q、K、V矩阵，并分割为三个部分
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 使用点积计算相关性，缩放后应用Softmax归一化。
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        # 将注意力权重与V矩阵相乘，得到加权聚合结果
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # 拼接多头结果并通过线性层映射回原始通道数
        res = self.output(res)
        # 残差魅力时刻
        res += x
        # 将结果还原为图像格式 (batch_size, channels, height, width)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


# DownBlock
# 即Unet Encoder中每一层的核心处理逻辑，Encoder的每一层都有2个DownBlock
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, attention: bool):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attention = nn.Identity()  # 不加注意力模块时直接空映射
        if attention:
            self.attention = AttentionBlock(out_channels)

    def forward(self, x, t):
        x=self.res(x, t)
        x=self.attention(x)


# UpBlock
# 即Unet Decoder中每一层的核心处理逻辑，Decoder的每一层都有3个UpBlock
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, attention: bool):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attention = nn.Identity()  # 不加注意力模块时直接空映射
        if attention:
            self.attention = AttentionBlock(out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attention(x)
        return x


# MiddleBlock
# 包含2个残差块和1个注意力块，旨在提升网络在低分辨率下的表现能力（低分辨率层通常是网络的瓶颈，包含了高层次的语义信息，因此需要更强的模型表达能力）
class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()

        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    # 定义数据流动过程
    def forward(self, x, t):
        # x = self.middle_block(x, t)
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x

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

# 下采样
# 通过步长为2的卷积操作，将输入特征图的空间尺寸（高度和宽度）减半，实现类似原Unet论文池化的效果
class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        return self.conv(x)


# 上采样
# 通过转置卷积（反卷积）操作，将输入特征图的空间尺寸扩大一倍，实现空间上采样。
class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        return self.conv(x)


# Unet
# 默认图像为RGB3通道
# 使用4层Decoder和Encoder，前2层无注意力，后2层注意力
# 初始卷积层通道数64，其后分别为 128, 128, 256
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # 获取步长向量
        self.time_embedding = TimeEmbedding(256)

        # 进入网络前先做一次初始卷积（使通道变为64）
        self.pre_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1)

        # Encoder4层，从上往下走，前2层不加注意力，每层包含2个downblock残差块用于特征提取
        self.encoder=nn.Sequential(
            DownBlock(in_channels=64, out_channels=64, time_channels=256, attention=False),  # encoder1_downblock1
            DownBlock(in_channels=64, out_channels=64, time_channels=256, attention=False),  # encoder1_downblock2
            Downsample(n_channels=64),  # down_sample1
            DownBlock(in_channels=64, out_channels=128, time_channels=256, attention=False),  # encoder2_downblock1
            DownBlock(in_channels=128, out_channels=128, time_channels=256, attention=False),  # encoder2_downblock2
            Downsample(n_channels=128),  # down_sample2
            DownBlock(in_channels=128, out_channels=256, time_channels=256, attention=True),  # encoder3_downblock1
            DownBlock(in_channels=256, out_channels=256, time_channels=256, attention=True),  # encoder3_downblock2
            Downsample(n_channels=256),  # down_sample3
            DownBlock(in_channels=256, out_channels=1024, time_channels=256, attention=True),  # encoder4_downblock1
            DownBlock(in_channels=1024, out_channels=1024, time_channels=256, attention=True)  # encoder4_downblock2
        )

        # 瓶颈层 MiddleBlock
        self.middle_block=MiddleBlock(n_channels=1024, time_channels=256)

        # Decoder4层，从下往上走，后2层不加注意力，每层包含3个upblock残差块
        # 看示意图，由于存在"Skip Connecting"结构，前面Encoder中相应层的结果会直接加过来输入，所以输入通道应是"和"的形式
        self.decoder=nn.Sequential(
            UpBlock(in_channels=1024 + 1024, out_channels=1024, time_channels=256, attention=True),  # decoder1_upblock1
            UpBlock(in_channels=1024 + 1024, out_channels=1024, time_channels=256, attention=True),  # decoder1_upblock2
            UpBlock(in_channels=1024 + 256, out_channels=256, time_channels=256, attention=True),  # decoder1_upblock3
            Upsample(n_channels=256),  # up_sample1
            UpBlock(in_channels=256 + 256, out_channels=256, time_channels=256, attention=True),  # decoder2_upblock1
            UpBlock(in_channels=256 + 256, out_channels=256, time_channels=256, attention=True),  # decoder2_upblock2
            UpBlock(in_channels=256 + 128, out_channels=128, time_channels=256, attention=True),  # decoder2_upblock3
            Upsample(n_channels=128),  # up_sample2
            UpBlock(in_channels=128 + 128, out_channels=128, time_channels=256, attention=False),  # decoder3_upblock1
            UpBlock(in_channels=128 + 128, out_channels=128, time_channels=256, attention=False),  # decoder3_upblock2
            UpBlock(in_channels=128 + 64, out_channels=64, time_channels=256, attention=False),  # decoder3_upblock3
            Upsample(n_channels=64),  # up_sample3
            UpBlock(in_channels=64 + 64, out_channels=64, time_channels=256, attention=False),  # decoder4_upblock1
            UpBlock(in_channels=64 + 64, out_channels=64, time_channels=256, attention=False),  # decoder4_upblock2
            UpBlock(in_channels=64 + 64, out_channels=64, time_channels=256, attention=False)  # decoder4_upblock3
        )

        # 最后一步还原为原输入图片的尺寸
        self.final_conv=nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=64),
            lambda x: x * torch.sigmoid(x),  # 即Swish()，一种平滑的非线性激活函数，增强非线性表达能力
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), padding=1)
        )


    # 模型有2个输入：
    # x：步长T下的图片数据
    # t：步长T
    def forward(self, x, t):
        # -----------------------
        # 预处理
        # -----------------------
        # 获取步长T的编码
        t = self.time_embedding(t)
        # 先把通道数卷到64再进入Encoder第一层
        x = self.pre_conv(x)

        # -----------------------
        # Encoder
        # -----------------------
        # 记录每经过Encoder中每一个Block后的结果，方便Decoder阶段的组装
        history = [x]
        for step in self.encoder:
            x= step(x, t)
            history.append(x)

        # -----------------------
        # MiddleBlock
        # -----------------------
        x = self.middle_block(x, t)

        # -----------------------
        # Decoder
        # -----------------------
        for step in self.decoder:
            # 若是上采样，直接做
            # 若是残差块，需要和Encoder中的数据concat一下再输入
            if isinstance(step, Upsample):
                x = step(x, t)
            else:
                x = step(torch.cat((x, history.pop()), dim=1), t)

        # -----------------------
        # 最终输出
        # -----------------------
        x = self.final_conv(x)
        return x


# ----------------------------------------------------------------------
# 4. 设置训练和验证方法、超参数
# ----------------------------------------------------------------------