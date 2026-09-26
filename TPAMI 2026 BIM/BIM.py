import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Cov_Block(nn.Module):
    """
    相似性感知加权模块（对应结构图Similarity-aware Weighting部分）
    核心：计算特征协方差矩阵，生成全局相似性权重，模仿大脑的特征匹配机制
    Args:
        dim: 输入通道数（双分支拼接后的总通道数）
    Input: 双分支拼接特征 [B, 2C, H, W]
    Output: 加权后的双分支拆分特征 [B,C,H,W] ×2
    """
    def __init__(self, dim):
        super().__init__()
        mid_dim = round(0.6 * dim)  # 隐藏层通道数，压缩60%降低计算量
        # MLP：将协方差矩阵映射为全局权重
        self.block = nn.Sequential(
            nn.Linear(dim, mid_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x):
        shortcut = x.clone()  # 保留原始特征用于加权
        # 维度重排：将空间维度展平，适配协方差计算
        x = rearrange(x, 'B C H W -> B (H W) C')
        # L2归一化，消除特征幅值差异对相似性计算的影响
        x = F.normalize(x, dim=-2)
        # 计算通道间协方差矩阵 [B, C, C]，衡量特征间的全局相似性
        cov = x.transpose(-2, -1) @ x
        # MLP生成全局相似性权重 [B, 1, 1]
        cov = self.block(cov).unsqueeze_(-1)
        # 全局加权增强：原始特征 × (1 + 权重)，保留原始信息的同时增强相似特征
        out = shortcut * (1 + cov)
        # 拆分为两个分支，适配后续高阶交互
        return out.chunk(2, dim=1)

class AttModule(nn.Module):
    """
    Brain-Inspired Module (BIM) 核心实现
    模仿大脑视觉皮层的信息处理流程：多尺度编码→相似性匹配→高阶交互
    对应结构图(c)的完整流程：Encoding → Similarity-aware Weighting → High-Order Interaction
    Args:
        dim: 输入/输出通道数
        bias: 卷积是否带偏置
    Input: 输入特征 [B, C, H, W]
    Output: 增强特征 [B, C, H, W]（与输入尺寸完全一致，即插即用）
    """
    def __init__(self, dim, bias):
        super().__init__()
        self.dim = dim
        # 阶段1：双分支多尺度编码（对应结构图Encoding部分）
        # 1×1卷积扩展通道为2倍，拆分为两个独立分支
        self.to_hidden = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        # 分支1：3×3深度卷积，提取局部细节特征（对应大脑V1区的小感受野神经元）
        self.to_hidden_dw1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 分支2：9×9深度卷积，提取全局结构特征（对应大脑V2区的大感受野神经元）
        self.to_hidden_dw2 = nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4, groups=dim, bias=bias)

        # 阶段2：相似性感知加权（对应结构图Similarity-aware Weighting部分）
        self.cov1 = Cov_Block(dim*2)

        # 阶段3：级联高阶交互（对应结构图High-Order Interaction部分）
        # 三级级联深度卷积+逐元素相乘，模拟神经元的级联激活与高阶信息整合
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # 输出投影：1×1卷积融合特征，恢复原始通道数
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 步骤1：双分支多尺度编码
        x_hidden = self.to_hidden(x)
        x1, x2 = x_hidden.chunk(2, dim=1)  # 拆分为两个分支
        x1 = self.to_hidden_dw1(x1)  # 3×3局部特征
        x2 = self.to_hidden_dw2(x2)  # 9×9全局特征

        # 步骤2：相似性感知全局加权
        x1, x2 = self.cov1(torch.cat([x1, x2], dim=1))

        # 步骤3：级联高阶交互（核心创新，模拟大脑神经元的级联激活）
        out = x1 * x2  # 一阶交互：局部×全局特征融合
        out = self.dw1(out) * x2  # 二阶交互：融合特征与全局特征交互
        out = self.dw2(out) * x1  # 三阶交互：融合特征与局部特征交互
        out = self.dw3(out) * x2  # 四阶交互：最终融合特征与全局特征交互

        # 步骤4：输出投影
        out = self.project_out(out)
        return out

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    model = AttModule(64, True).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)