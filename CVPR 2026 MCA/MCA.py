import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    """
    MCA: Multi-scale Color Attention 多尺度颜色注意力模块
    完整对应结构图右下角MCA模块全流程：Q/K/V/A四分支投影 → 光照先验融合 → 锚点级联注意力计算 → 输出投影
    核心特性：光照先验引导、低计算量通道级注意力、多尺度颜色特征捕捉
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 可学习双温度系数，分别控制两次注意力的分布平滑度，提升训练稳定性
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 【对应结构图Q分支】Query投影：分组深度可分离卷积+降采样，捕捉空间位置信息
        # 分组卷积保证通道独立性，stride=2实现多尺度降采样，reflect padding避免边缘信息丢失
        self.q_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect',
            groups=dim, bias=bias
        )

        # 【对应结构图K分支】Key投影：标准卷积+降采样，捕捉通道间颜色关联信息
        self.k_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect',
            bias=bias
        )

        # 【对应结构图V分支】Value投影：1×1卷积+光照先验融合，承载最终输出的特征信息
        # 与光照估计器输出的illu_feat逐元素相乘，注入颜色校正物理先验
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 【对应结构图A分支】Anchor锚点投影：降采样分组卷积+通道压缩，作为两次注意力的中间锚点
        # 实现通道维度降维，大幅降低注意力计算量，同时压缩空间冗余信息
        self.a_proj = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=3,
                padding=1, stride=2, padding_mode='reflect',
                groups=dim, bias=bias
            ),
            nn.Conv2d(dim, dim // 2, kernel_size=1)
        )

        # 输出投影：将注意力输出映射回原始通道维度
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        """
        前向主流程
        :param x: 输入特征 [B, C, H, W]，来自光谱编码器/颜色变换器的特征
        :param illu_feat: 光照先验特征 [B, C, H, W]，来自Illumination Estimator光照估计器（对应结构图V分支的元素乘）
        :return: 注意力增强后的特征 [B, C, H, W]，与输入同尺寸
        """
        b, c, h, w = x.shape
        # 步骤1：四分支并行投影，对应结构图Q/K/V/A四个卷积分支
        q = self.q_proj(x)  # Query: [B, C, H/2, W/2]
        k = self.k_proj(x)  # Key: [B, C, H/2, W/2]
        v = self.v_proj(x) * illu_feat  # Value: 融合光照先验，[B, C, H, W]
        a = self.a_proj(x)  # Anchor锚点: [B, C/2, H/2, W/2]

        # 步骤2：维度重整，适配多头注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 步骤3：归一化，保证注意力计算的数值稳定性
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = torch.nn.functional.normalize(a, dim=-1)

        # 步骤4：锚点级联注意力计算 对应结构图中两次矩阵乘法(⊗)
        # 第一次注意力：Query与锚点Anchor计算注意力，捕捉空间-锚点关联
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_a = attn_a.softmax(dim=-1)
        # 第二次注意力：锚点Anchor与Key计算注意力，捕捉锚点-通道关联
        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        attn_k = attn_k.softmax(dim=-1)

        # 步骤5：注意力加权，级联融合Value特征
        out_v = (attn_k @ v)
        out = (attn_a @ out_v)

        # 步骤6：维度还原与输出投影
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# 模块测试代码
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    illu_feat = torch.randn(1, 64, 32, 32).to(device)
    model = Attention(64, 4, False).to(device)
    y = model(x, illu_feat)
    print("微信公众号：十小大的底层视觉工坊")
    print("VX: shixiaodayyds, 备注【即插即用】添加交流群")
    print("小红书：十小大，主页获取往期全部即插即用代码，200+大合集")
    print("输入特征维度：", x.shape)
    print("输入先验特征维度：", illu_feat.shape)
    print("输出特征维度：", y.shape)