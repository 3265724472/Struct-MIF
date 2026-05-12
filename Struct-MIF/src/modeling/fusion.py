import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    特征融合层

    作用:
    将来自不同模态的特征 (序列流 ESM, 先验流 3Di) 进行拼接和降维/升维，
    为 GVP 准备初始的标量特征。
    """

    def __init__(self, esm_dim, struct_dim, output_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(esm_dim + struct_dim)
        self.proj = nn.Sequential(
            nn.Linear(esm_dim + struct_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, esm_feats, struct_feats):
        """
        Args:
            esm_feats: [N, 1280]
            struct_feats: [N, 128]
        Returns:
            fused_feats: [N, output_dim]
        """
        # 简单的拼接策略
        concat = torch.cat([esm_feats, struct_feats], dim=-1)
        normed = self.norm(concat)
        return self.proj(normed)