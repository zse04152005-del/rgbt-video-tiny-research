import torch.nn as nn
import torch
import torch.nn.functional as F


class ModulatedSpatialSpectralFusion(nn.Module):
    """
    Spatial-Channel Modulated Fusion（空间-通道调制融合，对应结构图(c) SCMF模块）
    核心架构：双输入特征拼接 → 空间/通道并行值分支 + 空间/通道并行门控分支 → 加权融合 → 残差输出
    针对多模态/多尺度特征融合，实现空间细节与通道语义的自适应精细化融合
    Args:
        c_in: 单路输入特征的通道数
    """
    def __init__(self, c_in):
        super().__init__()
        # ========== 空间值分支（对应结构图Spatial Modulation的Value路径）==========
        # 深度卷积提取空间细节，1×1卷积压缩通道，输出空间维度的融合值特征
        self.value_spatial = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in * 2, kernel_size=3, padding=1, groups=c_in * 2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(c_in * 2, c_in, kernel_size=1, bias=False),
            nn.LeakyReLU()
        )

        # ========== 通道值分支（对应结构图Channel Modulation的Value路径）==========
        # 1×1点卷积混合通道信息，输出通道维度的融合值特征
        self.value_spectral = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in, kernel_size=1, bias=False),
        )

        # ========== 空间门控分支（对应结构图Spatial Modulation的Weight路径）==========
        # 3×3卷积生成逐像素空间权重，Sigmoid归一化到0~1，自适应调节空间融合强度
        self.spatial_attention_head = nn.Sequential(
            nn.Conv2d(c_in * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # ========== 通道门控分支（对应结构图Channel Modulation的Weight路径）==========
        # 全局池化+1×1卷积生成逐通道权重，Sigmoid归一化到0~1，自适应调节通道融合强度
        self.spectral_attention_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in * 2, c_in, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fea, fea_en):
        """
        Args:
            fea: 主路径特征 (B, C, H, W)，对应结构图F_dec
            fea_en: 辅助特征 (B, C, H, W)，对应结构图F_enc
        Returns:
            output: 融合后特征 (B, C, H, W)，与主路径尺寸一致
        """
        B, C, H, W = fea.shape
        # 尺寸对齐：辅助特征分辨率不一致时双线性插值对齐
        if fea_en.shape[2:] != (H, W):
            fea_en = F.interpolate(fea_en, size=(H, W), mode="bilinear", align_corners=False)

        # 步骤1：两路特征通道拼接，作为融合输入
        x = torch.cat([fea, fea_en], dim=1)  # (B, 2C, H, W)

        # 步骤2：双值分支并行提取融合特征
        value_s = self.value_spatial(x)  # 空间值特征，侧重细节纹理融合
        value_p = self.value_spectral(x)  # 通道值特征，侧重语义信息融合

        # 步骤3：双门控分支生成自适应权重
        w_spatial = self.spatial_attention_head(x)  # 空间权重 (B, 1, H, W)
        w_spectral = self.spectral_attention_head(x)  # 通道权重 (B, C, 1, 1)

        # 步骤4：加权融合：空间支路按像素加权，通道支路按通道加权，逐元素相加
        fused_out = (value_s * w_spatial) + (value_p * w_spectral)

        # 步骤5：残差连接，保留主路径原始特征，实现增量式融合
        output = fused_out + fea
        return output


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn(1, 64, 32, 32).to(device)
    x2 = torch.randn(1, 64, 32, 32).to(device)
    model = ModulatedSpatialSpectralFusion(64).to(device)
    y = model(x1, x2)
    print("输入特征维度：", x1.shape)
    print("输入特征维度：", x2.shape)
    print("输出特征维度：", y.shape)