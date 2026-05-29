import torch
import torch.nn as nn
from torch_geometric.data import Batch
# 🔴 [新增] 用于动态计算 K 近邻
from torch_cluster import knn_graph

from .esm_wrapper import ESMWrapper
from .gvp_encoder import GVPGraphEncoder
from .fusion import FeatureFusion
from .gnn_adapters import GCNEncoder, GATEncoder, EGNNEncoder


class StructMIF(nn.Module):
    def __init__(
            self,
            esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
            gvp_node_in_dim: int = 1408,
            gvp_node_out_dim: int = 128,  # 🔴 控制隐层维度
            gvp_layers: int = 6,
            struct_vocab_size: int = 21,
            struct_embed_dim: int = 128,
            dropout: float = 0.1,
            ablation_mode: str = "none",
            gnn_type: str = "gvp",
            top_k: int = 0  # 🔴 [新增] 默认为0，表示使用数据自带的边；如果>0，则动态重构
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.gnn_type = gnn_type.lower()
        self.top_k = top_k

        # 1. Semantic Tower
        self.esm_encoder = ESMWrapper(esm_model_name)
        self.esm_hidden_dim = self.esm_encoder.hidden_size

        # 2. Prior Tower
        self.struct_embed_dim = struct_embed_dim
        self.struct_embedding = nn.Embedding(struct_vocab_size, struct_embed_dim)

        # 维度计算
        fusion_esm_dim = self.esm_hidden_dim
        fusion_struct_dim = struct_embed_dim
        if ablation_mode == "no_esm": fusion_esm_dim = 0
        if ablation_mode == "no_3di": fusion_struct_dim = 0

        # 3. Fusion Layer
        self.fusion_module = FeatureFusion(
            esm_dim=self.esm_hidden_dim,
            struct_dim=fusion_struct_dim,
            output_dim=gvp_node_in_dim,
            dropout=dropout
        )

        # 4. Geometric Engine
        print(
            f"🏗️ Building Geometric Engine: {self.gnn_type.upper()} | Dim: {gvp_node_out_dim} | K: {top_k if top_k > 0 else 'Fixed'}")

        if self.gnn_type == "gvp":
            self.encoder = GVPGraphEncoder(
                node_in_dim=gvp_node_in_dim,
                node_out_dim=gvp_node_out_dim,  # 🔴 使用传入的维度
                n_layers=gvp_layers,
                dropout=dropout
            )
        elif self.gnn_type == "gcn":
            self.encoder = GCNEncoder(
                node_in_dim=gvp_node_in_dim,
                node_out_dim=gvp_node_out_dim,  # 🔴 使用传入的维度
                n_layers=gvp_layers,
                dropout=dropout
            )
        elif self.gnn_type == "gat":
            self.encoder = GATEncoder(
                node_in_dim=gvp_node_in_dim,
                node_out_dim=gvp_node_out_dim,  # 🔴 使用传入的维度
                n_layers=gvp_layers,
                dropout=dropout
            )
        elif self.gnn_type == "egnn":
            self.encoder = EGNNEncoder(
                node_in_dim=gvp_node_in_dim,
                node_out_dim=gvp_node_out_dim,  # 🔴 使用传入的维度
                n_layers=gvp_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

        # 5. Head
        self.head = nn.Sequential(
            nn.LayerNorm(gvp_node_out_dim),  # 🔴 这里的归一化也要匹配维度
            nn.Linear(gvp_node_out_dim, self.esm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.esm_hidden_dim, 33)
        )

    def _rebuild_graph(self, batch):
        """
        🔴 [新增] 动态重构图逻辑
        利用 batch.pos 和 knn_graph 重新计算 edge_index
        """
        if not hasattr(batch, 'pos') or batch.pos is None:
            return batch  # 没坐标就算了

        # batch.batch 用于区分同一个 Batch 里不同的蛋白质
        # k=self.top_k
        new_edge_index = knn_graph(batch.pos, k=self.top_k, batch=batch.batch)
        batch.edge_index = new_edge_index

        # 重新计算边向量 (因为边变了，向量必须跟着变)
        row, col = new_edge_index
        coord_diff = batch.pos[col] - batch.pos[row]
        batch.edge_vectors = coord_diff.unsqueeze(1)  # [E, 1, 3]

        return batch

    def forward(self, batch: Batch):
        # 🔴 [关键] 如果设置了 top_k，先重构图
        if self.top_k > 0:
            batch = self._rebuild_graph(batch)

        # Step 1: ESM
        esm_out = self.esm_encoder(batch.input_ids, batch.esm_attention_mask)
        valid_mask = batch.attention_mask.bool()
        flat_esm_feats = esm_out[valid_mask]

        # Step 2: 3Di
        if self.ablation_mode == "no_3di":
            flat_3di_feats = torch.zeros((flat_esm_feats.size(0), 0), device=flat_esm_feats.device)
        else:
            flat_3di_feats = self.struct_embedding(batch.x_3di)

        if self.ablation_mode == "no_esm":
            flat_esm_feats = torch.zeros((flat_3di_feats.size(0), 0), device=flat_3di_feats.device)

        # Step 3: Fusion
        scalar_input = self.fusion_module(flat_esm_feats, flat_3di_feats)

        # Step 4: GNN Forward
        if self.gnn_type == "gvp":
            out = self.encoder(
                h_scalar=scalar_input,
                h_vector=batch.node_vectors,
                edge_index=batch.edge_index,
                edge_vector=batch.edge_vectors
            )
        elif self.gnn_type == "egnn":
            out = self.encoder(
                h=scalar_input,
                edge_index=batch.edge_index,
                pos=batch.pos
            )
        else:
            # GCN / GAT
            out = self.encoder(
                h=scalar_input,
                edge_index=batch.edge_index
            )

        # Step 5: Head
        logits = self.head(out)
        return logits