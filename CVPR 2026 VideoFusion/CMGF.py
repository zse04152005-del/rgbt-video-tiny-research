import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# -------------------------- 维度变换工具函数 --------------------------
def to_3d(x):
    """将4D图像特征[B,C,H,W]转为3D序列特征[B,N,C]，适配LayerNorm和注意力计算"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    """将3D序列特征还原为4D图像特征，适配卷积操作"""
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# -------------------------- 适配图像的LayerNorm模块 --------------------------
class BiasFree_LayerNorm(nn.Module):
    """无偏置层归一化，适配图像特征的通道级归一化"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置层归一化，Transformer标准归一化实现"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """图像特征统一LayerNorm封装，支持带偏置/无偏置两种模式，适配4D图像输入"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# -------------------------- 核心跨模态交叉注意力模块 对应结构图(c)核心计算单元 --------------------------
class Cross_Attention(nn.Module):
    """
    高效通道级跨模态交叉注意力
    对应结构图(c)中的Conv、q/k/v生成、trans.矩阵乘法、FFN前馈流程
    创新点：将空间注意力转为通道注意力，计算量从O((HW)²)降至O(C²)，适配高分辨率多模态融合
    """
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习温度系数，控制注意力分布
        self.norm = LayerNorm(dim, LayerNorm_type)  # 预归一化
        # Q/K/V生成：1×1卷积+深度可分离卷积，轻量化同时保留空间细节
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 输出投影

    def forward(self, x_A, x_B):
        """
        跨模态注意力前向
        :param x_A: Query特征（双模态融合特征Q）
        :param x_B: Key-Value特征（红外/可见光模态特征）
        :return: 模态引导后的增强特征
        """
        b, c, h, w = x_A.shape
        # Q/K/V生成与深度卷积增强
        q = self.q_dwconv(self.q(x_A))
        kv = self.kv_dwconv(self.kv(x_B))
        k, v = kv.chunk(2, dim=1)
        # 维度重整为多头格式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # L2归一化提升注意力稳定性
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 通道级注意力计算：C×C相似度矩阵，替代传统HW×HW空间注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 注意力加权Value特征
        out = (attn @ v)
        # 维度还原与输出投影
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# -------------------------- 门控前馈网络GDFN 对应结构图(c)中的FFN模块 --------------------------
class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)，来自Restormer，门控机制过滤无效特征"""
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # 门控融合，过滤无效特征
        x = self.project_out(x)
        return x

# -------------------------- 跨模态Transformer块 对应结构图(c)完整Transformer单元 --------------------------
class Cross_TransformerBlock(nn.Module):
    """
    预归一化跨模态Transformer块
    结构：LN → 跨模态注意力 → 残差连接 → LN → FFN → 残差连接
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(Cross_TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, num_heads, bias, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, supple):
        x = x + self.attn(self.norm1(x), self.norm1(supple))  # 注意力残差连接
        x = x + self.ffn(self.norm2(x))  # FFN残差连接
        return x

# -------------------------- CMGF主模块 对应结构图(c) Complete Modality-guided Fusion Module --------------------------
class FusionModule(nn.Module):
    """
    CMGF: Complete Modality-guided Fusion Module 完整模态引导融合模块
    完整对应结构图(c)全流程，是结构图(a)中时空协同视频融合网络的核心融合单元
    核心功能：双向模态引导的跨模态融合，同时保留红外的结构热目标与可见光的纹理细节
    """
    def __init__(self,
                 in_channels=192,
                 num_heads=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 embed_dim=64,
                 group=4):
        super(FusionModule, self).__init__()
        # 双路跨模态Transformer，分别实现红外引导、可见光引导
        self.cross_transformer = Cross_TransformerBlock(
            dim=in_channels,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            embed_dim=embed_dim,
            group=group)

    def forward(self, ir, vi):
        """
        前向主流程
        :param ir: 红外模态特征 [B,C,H,W]，对应结构图F^t_ir
        :param vi: 可见光模态特征 [B,C,H,W]，对应结构图F^t_vi
        :return: 融合后的特征 [B,C,H,W]，与输入同尺寸
        """
        # 校验双模态特征尺寸一致
        assert ir.shape == vi.shape, "ir and vi must have the same shape."
        # 构建融合Query：双模态特征加和，同时包含两个模态的基础信息
        Q = ir + vi
        # 红外模态引导的交叉注意力：强化热目标、结构轮廓信息
        fusion_ir = self.cross_transformer(Q, ir)
        # 可见光模态引导的交叉注意力：强化纹理、细节、亮度信息
        fusion_vi = self.cross_transformer(Q, vi)
        # 双向引导特征融合，输出最终结果
        fusion = fusion_ir + fusion_vi
        return fusion


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn(1, 64, 32, 32).to(device)
    x2 = torch.randn(1, 64, 32, 32).to(device)
    model = FusionModule(64, 4, 2).to(device)
    y = model(x1,x2)
    print("微信公众号：十小大的底层视觉工坊")
    print("VX: shixiaodayyds, 备注【即插即用】添加交流群")
    print("小红书：十小大，主页获取往期全部即插即用代码，200+大合集")
    print("输入特征1维度：", x1.shape)
    print("输入特征2维度：", x2.shape)
    print("输出特征维度：", y.shape)