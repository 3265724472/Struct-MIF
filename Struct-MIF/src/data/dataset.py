import os
import glob
import torch
import logging
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StructMIFDataset(Dataset):
    """
    Struct-MIF Dataset (Training Mode) v2
    修复：在 apply_masking 后，裁掉 Labels 的首尾，以匹配模型的 GVP 输出。
    """

    def __init__(self, root_dir, tokenizer_name, max_len=1024, mask_prob=0.15):
        self.root_dir = root_dir
        self.max_len = max_len
        self.mask_prob = mask_prob

        # 1. 加载 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_name}. Error: {e}")
            raise

        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            logger.warning("Tokenizer doesn't have a mask_token_id! Defaulting to 32.")
            self.mask_token_id = 32

        self.vocab_size = self.tokenizer.vocab_size
        self.vocab_3di = "ACDEFGHIKLMNPQRSTVWY"
        self.char_to_int_3di = {c: i for i, c in enumerate(self.vocab_3di)}

        # 2. 扫描并过滤文件
        raw_files = glob.glob(os.path.join(root_dir, "*.pt"))
        if len(raw_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {root_dir}")

        logger.info(f"Scanning {len(raw_files)} files in {root_dir}...")
        self.file_list = []
        for f in tqdm(raw_files, desc="Filtering by length"):
            try:
                # 简单读取 header 检查长度
                data = torch.load(f)
                if data.num_nodes <= self.max_len:
                    self.file_list.append(f)
            except Exception:
                continue

        logger.info(f"Dataset loaded successfully. Valid samples: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def apply_masking(self, input_ids):
        """
        标准的 BERT/ESM 掩码策略
        """
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # 排除特殊字符
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 没选中的不计算 loss

        # 80% 替换为 <mask >
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% 替换为随机词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels

    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = torch.load(path)
        seq = data.seq

        # 1. 原始序列编码
        token_out = self.tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
        raw_input_ids = token_out["input_ids"].squeeze(0)

        # 2. 应用 Masking
        masked_input, masked_labels = self.apply_masking(raw_input_ids.clone())

        data.input_ids = masked_input  # 包含 CLS/EOS，长度 L

        # [关键修复]：裁掉 labels 的首尾，使其长度等于 Graph Nodes (L-2)
        # 因为模型只预测中间的残基，所以 labels 也只能给中间的
        # masked_labels 的结构是 [<cls>, aa, aa, ..., <eos>]
        # 我们取 [1:-1]
        if masked_labels.size(0) > 2:
            data.y = masked_labels[1:-1]
        else:
            # 防御性逻辑 (虽然不太可能出现长度<2的蛋白)
            data.y = masked_labels

        # 3. 设置 Masks
        data.esm_attention_mask = torch.ones_like(masked_input)

        align_mask = torch.zeros_like(masked_input)
        if len(align_mask) > 2:
            align_mask[1:-1] = 1
        data.attention_mask = align_mask

        # 4. 3Di 序列
        if hasattr(data, 'seq_3di') and data.seq_3di is not None:
            indices = [self.char_to_int_3di.get(c, 0) for c in data.seq_3di]
            data.x_3di = torch.tensor(indices, dtype=torch.long)
        else:
            data.x_3di = torch.zeros(data.num_nodes, dtype=torch.long)

        # 5. Edge Vectors
        if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
            row, col = data.edge_index
            vec = data.pos[col] - data.pos[row]
            data.edge_vectors = vec.unsqueeze(1)
        else:
            num_edges = data.edge_index.shape[1]
            data.edge_vectors = torch.zeros(num_edges, 1, 3)

        # 6. Node Vectors
        if hasattr(data, 'node_vectors'):
            n_vec = data.node_vectors[:, 0, :]
            c_vec = data.node_vectors[:, 1, :]
            cross_vec = torch.cross(n_vec, c_vec, dim=1)
            data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

        return data