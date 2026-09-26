import math
import torch
import torch.nn as nn
from einops import rearrange

class DecayPos1d_2(nn.Module):
    """
    可学习温度的一维衰减位置编码（用于光谱注意力分支）
    核心：模拟光谱维度的相邻波段相关性，距离越远衰减越大
    Args:
        embed_dim: 总嵌入维度
        num_heads: 注意力头数
        initial_value: 初始衰减值
        heads_range: 衰减值范围
        init_tau: 初始温度系数
    """
    def __init__(self, embed_dim, num_heads, initial_value, heads_range, init_tau=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.init_tau = nn.Parameter(torch.tensor(3.0))  # 可学习温度系数
        # 预计算每个头的基础衰减值
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int):
        """生成光谱维度的衰减掩码"""
        index = torch.arange(l, device=self.decay.device)
        # 计算波段间的绝对距离
        mask = (index[:, None] - index[None, :]).abs()
        # 可学习温度缩放
        mask = (mask / self.init_tau).exp()
        return mask * self.decay[:, None, None]  # [heads, l, l]

    def forward(self, slen):
        return self.generate_1d_decay(slen)

def gaussian_spatial_position(h, w, sigma=1.0, device=None):
    """
    生成高斯空间位置权重矩阵（用于空间注意力分支）
    核心：模拟空间维度的局部相关性，距离中心越近权重越高
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    center_h, center_w = h // 2, w // 2
    y_coords = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1)
    x_coords = torch.arange(w, dtype=torch.float32, device=device).view(1, -1)
    # 计算每个像素到中心的欧氏距离，生成高斯权重
    dist_sq = (y_coords - center_h) ** 2 + (x_coords - center_w) ** 2
    return torch.exp(-dist_sq / (2 * sigma ** 2))

class CenterPixelAttention(nn.Module):
    """
    Central Pixel-guided Dual-path Multihead Attention (CPDMA) 核心实现
    对应结构图(b)的双路径注意力：空间注意力+光谱注意力，中心像素引导双分支
    专为高光谱图像设计，同时建模空间和光谱维度的依赖关系
    Args:
        dim: 输入/输出通道数（光谱维度）
        heads: 注意力头数
        dim_heads: 每个头的维度
        dropout: dropout概率
    """
    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.dim = dim
        self.dim_heads = dim_heads
        self.scale = dim_heads ** -0.5

        # ====================== 第一阶段：中心像素空间注意力 ======================
        # 对应结构图(b)上方的Spatial Multi-head Attention
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias=False)  # 仅生成K和V
        self.to_q_center = nn.Linear(dim, inner_dim, bias=False)  # 中心像素生成Q

        # ====================== 第二阶段：光谱注意力 ======================
        # 对应结构图(b)下方的Spectral Multi-head Attention
        self.q_lr_self = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # 深度卷积空间增强
            nn.Conv2d(dim, inner_dim, kernel_size=1)  # 点卷积通道变换
        )
        self.self_attn_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        self.self_attn_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )

        # ====================== 先验知识模块 ======================
        self.realPos = DecayPos1d_2(dim, heads, 2, 4)  # 光谱衰减位置编码
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)  # 高斯先验权重

        # ====================== 输出层 ======================
        self.final_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        前向传播，严格对应结构图流程
        Args:
            x: 输入特征 [B, N, D]，N=H×W为空间像素数，D为光谱通道数
        Returns:
            增强特征 [B, N, D]
        """
        b, n, d = x.shape
        h = self.heads
        h1 = w1 = int(math.sqrt(n))  # 空间尺寸H×W
        center_idx = n // 2  # 中心像素索引

        # 预计算高斯空间位置先验 [1,1,1,N]
        g_pos = gaussian_spatial_position(h1, w1, device=x.device)
        g_pos = g_pos.view(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # ====================== 第一阶段：中心像素空间注意力 ======================
        # 1. 提取中心像素作为Q（核心创新：Q仅1个，计算量从O(n²)→O(n)）
        center_pixel = x[:, center_idx:center_idx + 1, :]  # [B,1,D]
        q = self.to_q_center(center_pixel)
        # 2. 所有像素生成K和V
        k, v = self.to_qkv(x).chunk(2, dim=-1)  # [B,N,D] ×2
        # 3. 重塑为多头格式
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)  # [B,H,1,d]
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)  # [B,H,N,d]
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)  # [B,H,N,d]
        # 4. 计算空间注意力权重
        q = q * self.scale
        attn_spatial = torch.einsum('bhqd,bhnd->bhqn', q, k)  # [B,H,1,N]
        # 5. 融合高斯空间先验（距离中心越近权重越高）
        attention_weights = attn_spatial + self.alpha * g_pos
        attention_weights = attention_weights.softmax(dim=-1).transpose(-1, -2)  # [B,H,N,1]
        # 6. 加权V得到空间注意力输出
        out_spatial = v * attention_weights  # [B,H,N,d]
        out_spatial = rearrange(out_spatial, 'b h n d -> b n (h d)')  # [B,N,D]

        # ====================== 第二阶段：光谱注意力 ======================
        # 1. 将特征重塑为2D空间格式 [B,D,H,W]
        out_2d = rearrange(x, 'b (h1 w1) d -> b d h1 w1', h1=h1, w1=w1)
        # 2. 卷积增强后提取中心像素作为光谱Q
        q_self = self.q_lr_self(out_2d)  # [B,inner_dim,H,W]
        center_h, center_w = h1 // 2, w1 // 2
        q_center = q_self[:, :, center_h, center_w].unsqueeze(1)  # [B,1,inner_dim]
        q_center = rearrange(q_center, 'b n (h d) -> b h d n', h=h)  # [B,H,d,1]
        # 3. 全局平均池化生成光谱K（捕捉全局光谱统计）
        k_self = self.self_attn_k(out_2d)
        k_self_avg = k_self.mean(dim=(2, 3)).unsqueeze(1)  # [B,1,inner_dim]
        k_self_avg = rearrange(k_self_avg, 'b n (h d) -> b h d n', h=h)  # [B,H,d,1]
        # 4. 生成光谱V
        v_self = self.self_attn_v(out_2d)
        v_self = rearrange(v_self, 'b (hd h_d) h1 w1 -> b hd h_d (h1 w1)', hd=h, h_d=self.dim_heads)  # [B,H,d,N]
        # 5. 计算光谱注意力权重 [B,H,d,d]
        spectral_attn = q_center @ k_self_avg.transpose(-1, -2)
        # 6. 融合光谱衰减位置编码（相邻波段相关性更强）
        realPos = self.realPos(self.dim_heads).unsqueeze(0)  # [1,H,d,d]
        spectral_attn = spectral_attn + realPos
        spectral_attn = spectral_attn.softmax(dim=-1)
        # 7. 加权V得到光谱注意力输出
        out_spectral = spectral_attn @ v_self  # [B,H,d,N]
        out_spectral = rearrange(out_spectral, 'b h d n -> b n (h d)')  # [B,N,D]

        # ====================== 双路径融合与输出 ======================
        # 对应结构图(c)的Feature Fusion：逐元素相乘融合空间和光谱信息
        out_fused = out_spatial * out_spectral
        # 最终线性投影与dropout
        out_final = self.final_out(out_fused)
        return out_final



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 32*32, 64).to(device)
    model = CenterPixelAttention(64, 4, 16, 0).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)