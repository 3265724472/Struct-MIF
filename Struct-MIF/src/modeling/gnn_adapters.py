import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing


class BaseGNNEncoder(nn.Module):
    def __init__(self, node_in_dim, node_out_dim, n_layers, dropout=0.1):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_out_dim = node_out_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # 降维投影
        self.input_proj = nn.Linear(node_in_dim, node_out_dim)


class GCNEncoder(BaseGNNEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convs = nn.ModuleList([
            GCNConv(self.node_out_dim, self.node_out_dim)
            for _ in range(self.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.node_out_dim) for _ in range(self.n_layers)
        ])

    def forward(self, h, edge_index, **kwargs):
        x = self.input_proj(h)
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in
        return x


class GATEncoder(BaseGNNEncoder):
    # 🔴 注意：heads 改为 1 以避免维度爆炸
    def __init__(self, heads=1, **kwargs):
        super().__init__(**kwargs)
        self.convs = nn.ModuleList([
            GATConv(self.node_out_dim, self.node_out_dim, heads=heads, concat=False)
            for _ in range(self.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.node_out_dim) for _ in range(self.n_layers)
        ])

    def forward(self, h, edge_index, **kwargs):
        x = self.input_proj(h)
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in
        return x


class SimpleEGNNLayer(MessagePassing):
    """
    简化版 EGNN Layer (E-GCL) - Invariant Mode
    """

    # 🔴 [修复点] 参数名改为 hidden_dim
    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__(aggr='add')

        # 🔴 [修复点] 绝对不要覆盖 self.node_dim
        # self.node_dim = hidden_dim  <-- 这一行是万恶之源，必须删掉！
        self.hidden_dim = hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        dist_sq = (pos_i - pos_j).pow(2).sum(dim=-1, keepdim=True)
        msg_input = torch.cat([x_i, x_j, dist_sq], dim=-1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1)) + x


class EGNNEncoder(BaseGNNEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleList([
            SimpleEGNNLayer(self.node_out_dim)
            for _ in range(self.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.node_out_dim) for _ in range(self.n_layers)
        ])

    def forward(self, h, edge_index, pos, **kwargs):
        x = self.input_proj(h)
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, pos, edge_index)
            x = norm(x)
        return x