import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


class StructMIFCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch_list):
        """
        Collator v3: 强制堆叠 ESM 输入，解决 PyG 自动 Concat 导致的维度错误。
        """
        processed_batch = []

        # ---------------------------------------------------
        # 第一步：截断 (Truncation)
        # ---------------------------------------------------
        for data in batch_list:
            L = len(data.input_ids)
            if L > self.max_length:
                # 截断 input_ids (保留 CLS, 截断中间, 补 EOS)
                cutoff = self.max_length - 1
                trunc_input = data.input_ids[:cutoff].clone()
                trunc_input[-1] = self.eos_token_id
                data.input_ids = trunc_input

                # 截断 y (labels)
                num_residues = self.max_length - 2
                if hasattr(data, 'y') and data.y is not None:
                    data.y = data.y[:num_residues]

                # 截断 Masks
                if hasattr(data, 'esm_attention_mask'):
                    data.esm_attention_mask = data.esm_attention_mask[:self.max_length]
                if hasattr(data, 'attention_mask'):
                    data.attention_mask = data.attention_mask[:self.max_length]

                # 截断 图节点特征
                if hasattr(data, 'x_3di'): data.x_3di = data.x_3di[:num_residues]
                if hasattr(data, 'node_vectors'): data.node_vectors = data.node_vectors[:num_residues]
                if hasattr(data, 'pos'): data.pos = data.pos[:num_residues]

                # 截断 Edge Index
                if hasattr(data, 'edge_index'):
                    mask_edges = (data.edge_index[0] < num_residues) & (data.edge_index[1] < num_residues)
                    data.edge_index = data.edge_index[:, mask_edges]
                    if hasattr(data, 'edge_vectors'):
                        data.edge_vectors = data.edge_vectors[mask_edges]
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        data.edge_attr = data.edge_attr[mask_edges]

                # 更新 num_nodes
                data.num_nodes = num_residues

            # 安全检查
            expected_nodes = len(data.input_ids) - 2
            if data.num_nodes != expected_nodes:
                data.num_nodes = expected_nodes
                limit = min(data.num_nodes, expected_nodes)
                if hasattr(data, 'x_3di'): data.x_3di = data.x_3di[:limit]

            processed_batch.append(data)

        # ---------------------------------------------------
        # 第二步：填充 (Padding)
        # ---------------------------------------------------
        batch_max_len = max([len(d.input_ids) for d in processed_batch])

        # 用列表收集 Pad 后的 Tensor，准备手动 Stack
        padded_input_ids = []
        padded_esm_mask = []
        padded_att_mask = []

        for data in processed_batch:
            curr_len = len(data.input_ids)
            pad_len = batch_max_len - curr_len

            # Pad input_ids
            # 注意：即使 pad_len=0 也要 pad (保持 tensor 类型一致)
            padded_input = F.pad(data.input_ids, (0, pad_len), value=self.pad_token_id)
            data.input_ids = padded_input  # 更新回 data 以防 PyG 需要检查
            padded_input_ids.append(padded_input)

            # Pad esm_mask (0)
            if hasattr(data, 'esm_attention_mask'):
                padded_esm = F.pad(data.esm_attention_mask, (0, pad_len), value=0)
                data.esm_attention_mask = padded_esm
                padded_esm_mask.append(padded_esm)

            # Pad att_mask (0)
            if hasattr(data, 'attention_mask'):
                padded_att = F.pad(data.attention_mask, (0, pad_len), value=0)
                data.attention_mask = padded_att
                padded_att_mask.append(padded_att)

        # ---------------------------------------------------
        # 第三步：混合堆叠 (Hybrid Stacking)
        # ---------------------------------------------------
        # 1. 让 PyG 处理图结构数据 (x_3di, edge_index 等会正确 Concat)
        batch = Batch.from_data_list(processed_batch)

        # 2. [核心修复] 手动覆盖 ESM 相关数据为 2D Stacked Tensor
        # 这样 input_ids 就会变成 [Batch, Max_Len] 而不是 [Batch * Max_Len]
        batch.input_ids = torch.stack(padded_input_ids, dim=0)

        if len(padded_esm_mask) > 0:
            batch.esm_attention_mask = torch.stack(padded_esm_mask, dim=0)

        if len(padded_att_mask) > 0:
            batch.attention_mask = torch.stack(padded_att_mask, dim=0)

        # 注意：batch.y (labels) 不需要手动 stack，
        # 因为 Loss 计算本身就需要 flatten 的 labels，PyG 的默认 Concat 行为正好符合要求。

        return batch