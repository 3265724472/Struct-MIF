import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# 依赖 gvp-pytorch 库
# GitHub: https://github.com/drorlab/gvp-pytorch
import gvp
from gvp import GVP, GVPConv, GVPConvLayer, LayerNorm


class GVPGraphEncoder(nn.Module):
    """
    GVP-GNN 几何编码器

    作用:
    接收融合后的标量特征 (ESM+3Di) 和 PDB 骨架向量特征，
    进行旋转等变 (Rotation-Equivariant) 的消息传递。
    """

    def __init__(
            self,
            node_in_dim: int,  # 输入标量维度 (来自 StructMIF 的 fusion_proj 输出, e.g., 1408)
            node_out_dim: int,  # 输出/隐藏层标量维度 (e.g., 128)
            node_vector_dim: int = 3,  # 输入向量通道数 (通常 3: Forward, Reverse, Sidechain)
            edge_vector_dim: int = 1,  # 边向量通道数 (通常 1: 相对位置向量)
            n_layers: int = 6,  # GVP 层数
            dropout: float = 0.1,
            vector_hidden_dim: int = 16  # 隐藏层向量通道数 (GVP 内部向量维度)
    ):
        super().__init__()

        # GVP 的输入/隐藏层配置元组 (scalar_dims, vector_dims)
        # 输入层配置
        self.in_dims = (node_in_dim, node_vector_dim)
        # 隐藏层配置
        self.hidden_dims = (node_out_dim, vector_hidden_dim)

        # 1. Embedding 层: 将输入特征映射到隐藏层维度
        # GVP 模块同时处理 (s, V) -> (s', V')
        self.W_in = nn.Sequential(
            LayerNorm(self.in_dims),
            GVP(self.in_dims, self.hidden_dims, activations=(None, None))
        )

        # 2. GVP-GNN 堆叠层
        # 使用 GVPConvLayer，它封装了 Message Passing + Dropout + Norm
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                GVPConvLayer(
                    self.hidden_dims,  # Node dims (s, V)
                    (0, edge_vector_dim),  # Edge dims (s, V) - 假设没有边标量，只有边向量
                    drop_rate=dropout,
                    activations=(F.relu, None),  # 标量用 ReLU，向量不做激活或用 Vector Gating
                    vector_gate=True  # 开启向量门控，增强几何表达能力
                )
            )

        # 3. 输出层 (可选)
        # 如果只需要标量输出用于预测氨基酸，这里其实不需要额外处理，
        # 因为最后一层 GVPConvLayer 已经输出了 (s_out, V_out)

    def forward(self, h_scalar, h_vector, edge_index, edge_vector):
        """
        Args:
            h_scalar: [Total_Nodes, node_in_dim] (ESM+3Di features)
            h_vector: [Total_Nodes, 3, 3] (PDB backbone vectors)
            edge_index: [2, Total_Edges]
            edge_vector: [Total_Edges, 1, 3] (Relative position vectors)

        Returns:
            scalar_out: [Total_Nodes, node_out_dim]
        """

        # 1. 初始特征打包
        # GVP 库要求输入为元组 (Scalar, Vector)
        # h_vector 需要 reshape 成 [N, node_vector_dim, 3] -> 已经在 Dataset 里做好了
        x = (h_scalar, h_vector)

        # 2. 输入映射
        x = self.W_in(x)

        # 3. 边特征打包
        # 假设没有边标量特征 (dummy edge scalar)，只有边向量
        # 创建 dummy edge scalar [E, 0]
        edge_scalar = torch.zeros((edge_index.size(1), 0), device=edge_index.device)
        edge_attrs = (edge_scalar, edge_vector)

        # 4. GVP 消息传递
        for layer in self.layers:
            x = layer(x, edge_index, edge_attrs)

        # 5. 提取标量部分作为输出
        # x 是 (scalar, vector)
        scalar_out, vector_out = x

        return scalar_out