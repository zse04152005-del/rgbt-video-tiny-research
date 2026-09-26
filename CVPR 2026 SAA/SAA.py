import torch
import torch.nn as nn
from torch.nn import functional as F


import torch
import torch.nn as nn
from torch.nn import functional as F

def cluster_and_merge(x, cluster_num, subsample_factor=4):
    """
    密度驱动的token聚类与合并（对应结构图中的Density-driven Token Aggregation, DTA）
    核心：根据特征密度自适应选择聚类中心，合并相似token生成紧凑的KV集合
    Args:
        x: 输入token序列 [B, N, C]
        cluster_num: 聚类中心数量（即输出token数）
        subsample_factor: 下采样因子，用于预筛选候选中心
    Returns:
        out: 聚合后的token序列 [B, cluster_num, C]
    """
    B, N, C = x.shape
    device = x.device
    K = cluster_num

    # 步骤1：特征归一化，用于余弦相似度计算
    x_norm = F.normalize(x, dim=-1)

    # 步骤2：分层下采样预筛选候选中心（对应结构图的Stratified Subsampling）
    S = min(N, max(2 * K, subsample_factor * K))  # 候选数≥2K，保证覆盖性
    samples_per_region = S // K
    sub_idx = []
    # 分区域均匀采样，避免局部聚集
    for i in range(K):
        start_idx = i * (N // K)
        end_idx = (i + 1) * (N // K) if i < K - 1 else N
        region_size = end_idx - start_idx
        n_samples = min(samples_per_region, region_size)
        if region_size > 0:
            region_perm = torch.randperm(region_size, device=device)[:n_samples]
            sub_idx.append(start_idx + region_perm)
    sub_idx = torch.cat(sub_idx)
    # 补全随机样本至S个
    if len(sub_idx) < S:
        remaining = S - len(sub_idx)
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        additional = all_idx[mask][torch.randperm((~mask).sum(), device=device)[:remaining]]
        sub_idx = torch.cat([sub_idx, additional])
    x_norm_sub = x_norm[:, sub_idx]  # [B, S, C]

    # 步骤3：密度引导的中心选择（对应结构图的Density-Guided Center Selection）
    # 3.1 计算候选间余弦相似度
    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)  # [B, S, S]
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)  # 排除自身
    # 3.2 计算局部密度：top-k相似度的均值
    k = min(K, S - 1)
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)
    density_sub = sim_topk_sub.mean(dim=-1)  # [B, S]
    density_sub += torch.rand_like(density_sub) * 1e-6  # 避免数值问题
    # 3.3 计算相对距离δ：到更高密度点的最大相似度的补
    mask_higher_density = (density_sub[:, None, :] > density_sub[:, :, None]).float()
    masked_sim_sub = sim_sub * mask_higher_density - 1e9 * (1.0 - mask_higher_density)
    max_sim_to_higher, _ = masked_sim_sub.max(dim=-1)
    delta_sub = 1.0 - max_sim_to_higher
    # 3.4 处理密度峰值点（无更高密度邻居）
    max_density_mask_sub = (mask_higher_density.sum(dim=-1) == 0)
    min_sim_global = sim_sub.min(dim=-1)[0]
    max_dist_global = 1.0 - min_sim_global
    delta_sub[max_density_mask_sub] = max_dist_global[max_density_mask_sub]
    delta_sub = torch.clamp(delta_sub, min=0.0)
    # 3.5 综合得分γ=ρ×δ，选择top-K作为聚类中心
    score_sub = density_sub * delta_sub
    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)
    center_idx = sub_idx[center_idx_in_sub]  # 映射回原始索引
    centers_norm = torch.gather(x_norm, 1, center_idx[..., None].expand(B, K, C))

    # 步骤4：token聚类与加权合并（对应结构图的weighted aggregation）
    # 4.1 计算每个token与中心的相似度，分配到最近簇
    sim_token_center = x_norm @ centers_norm.transpose(1, 2)  # [B, N, K]
    assign_idx = sim_token_center.argmax(dim=-1)  # [B, N]
    # 4.2 加权平均合并簇内token（使用原始特征保证质量）
    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)
    cluster_counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)
    out = torch.einsum("bnc,bnk->bkc", x, one_hot) / cluster_counts.transpose(1, 2)
    return out

class SAA(nn.Module):
    """
    Selective Aggregation Attention (选择性聚合注意力)
    核心：通过密度驱动的token聚合生成紧凑KV集合，用交叉注意力替代全局自注意力
    复杂度从O(n²)降至O(nk)（k≈3%n），输入输出尺寸完全一致
    Args:
        dim: 输入/输出通道数
        num_heads: 注意力头数
        qkv_bias: QKV是否带偏置
        qk_scale: 注意力缩放因子
        attn_drop: 注意力dropout
        proj_drop: 输出dropout
        c_ratio: Q/K通道压缩比，默认0.5
        M: 聚合token占比，默认0.03（3%）
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5, M=0.03):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.cr = int(dim * c_ratio)  # Q/K压缩后的通道数
        self.scale = qk_scale or (self.cr // num_heads) ** -0.5
        self.M = M  # 聚合token占原始token的比例

        # QKV投影：Q/K通道压缩，V保持原通道
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 步骤1：密度驱动token聚合，生成紧凑KV集合
        NF = int(self.M * N)  # 聚合后的token数（默认3%N）
        T_avg = cluster_and_merge(x, NF)

        # 步骤2：范数保持机制（解决聚合后特征幅值下降问题）
        max_norm = torch.norm(x, dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        avg_norm = torch.norm(T_avg, dim=-1, keepdim=True)
        epsilon = 1e-6
        mask = avg_norm > epsilon
        scaled = (T_avg / (avg_norm + epsilon)) * max_norm
        KV_comp = torch.where(mask, scaled, T_avg)  # [B, k, C]
        K_size = KV_comp.shape[1]

        # 步骤3：通道压缩交叉注意力（对应结构图右侧）
        # Q：原始所有token，通道压缩
        q = self.q(x).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        # K/V：聚合后的token，K通道压缩，V保持原通道
        k = self.k(KV_comp).reshape(B, K_size, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(KV_comp).reshape(B, K_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, k]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和与输出投影
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 32*32, 64).to(device)
    model = SAA(64).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)