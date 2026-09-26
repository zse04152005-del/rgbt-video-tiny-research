import torch
import torch.nn as nn

# -------------------------- 通道注意力模块 对应结构图(b)中的CA模块 --------------------------
class ChannelAttention(nn.Module):
    """
    双池化通道注意力，对加权后的主模态特征做通道维度的自适应增强
    核心：通过全局平均+最大池化捕捉通道全局统计特性，学习不同通道的重要性权重
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局池化，压缩空间维度
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 通道压缩-还原MLP，降低参数量
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双池化分支分别生成通道权重，相加融合
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

# -------------------------- 空间注意力模块 对应结构图(b)中的SA模块 --------------------------
class SpatialAttention(nn.Module):
    """
    空间注意力模块，对通道增强后的特征做空间维度的自适应增强
    核心：通过通道维度的平均+最大池化，捕捉空间位置的显著性，生成空间权重
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 3×3卷积融合通道池化后的特征，生成空间权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的平均+最大池化，压缩通道维度
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后卷积生成空间权重
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

# -------------------------- CmDRM核心模块 对应结构图(b) Cross-modal Differential Reinforcement全流程 --------------------------
class CrossModalAggregation(nn.Module):
    """
    CmDRM: Cross-modal Differential Reinforcement Module 跨模态差分增强模块
    完整对应结构图(b)全流程，是结构图(a)时空协同视频融合网络的核心跨模态增强单元
    核心功能：自适应学习双模态权重，强化模态间的差分互补信息，抑制冗余信息，通过通道-空间注意力精细化增强主模态特征
    """
    def __init__(self, in_channels):
        super(CrossModalAggregation, self).__init__()
        # 1×1卷积：从拼接的双模态特征中学习两个模态的自适应权重
        self.conv = nn.Conv2d(in_channels * 2, 2, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 全局平均池化：生成模态级的全局权重
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 通道-空间串行注意力，对主模态做精细化增强
        self.ChannelAttention = ChannelAttention(in_channels)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x_main, x_supple):
        """
        前向主流程
        :param x_main: 主模态特征（如红外特征，对应结构图F^t_ir），核心结构/热目标信息载体
        :param x_supple: 辅助模态特征（如可见光特征，对应结构图F^t_vi），细节纹理信息载体
        :return: 差分增强后的融合特征，与输入同尺寸
        """
        # -------------------------- 步骤1：双模态拼接与权重生成 对应结构图(b) C拼接+Conv --------------------------
        x = torch.cat([x_main, x_supple], dim=1)
        x = self.conv(x)
        # 拆分权重：分别对应主模态、辅助模态的自适应权重
        w_main, w_supple = x[:, 0:1, :, :], x[:, 1:2, :, :]
        # Sigmoid归一化到0-1，保证权重的有效性
        w_main = self.sigmoid(w_main)
        w_supple = self.sigmoid(w_supple)

        # -------------------------- 步骤2：全局池化与模态加权 对应结构图(b) AP模块 --------------------------
        # 全局池化生成模态级全局权重，实现差分增强：强化互补信息，抑制冗余信息
        w_main = self.pool(w_main)
        w_supple = self.pool(w_supple)
        # 权重加权双模态特征，实现模态级的差分筛选
        g_inp1 = w_main * x_main
        g_inp2 = w_supple * x_supple

        # -------------------------- 步骤3：主模态通道-空间精细化增强 对应结构图(b) CA+SA模块 --------------------------
        # 通道注意力增强：强化主模态的关键通道
        x_main_CA = self.ChannelAttention(g_inp1) * g_inp1
        # 空间注意力增强：强化主模态的关键空间区域
        x_main_CA_SA = self.SpatialAttention(x_main_CA) * x_main_CA

        # -------------------------- 步骤4：双模态融合输出 --------------------------
        fuse = x_main_CA_SA + g_inp2
        return fuse


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn(1, 64, 32, 32).to(device)
    x2 = torch.randn(1, 64, 32, 32).to(device)
    model = CrossModalAggregation(64).to(device)
    y = model(x1, x2)
    print("微信公众号：十小大的底层视觉工坊")
    print("VX: shixiaodayyds, 备注【即插即用】添加交流群")
    print("小红书：十小大，主页获取往期全部即插即用代码，200+大合集")
    print("输入特征1维度：", x1.shape)
    print("输入特征2维度：", x2.shape)
    print("输出特征维度：", y.shape)