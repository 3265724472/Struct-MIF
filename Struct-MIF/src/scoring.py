import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


class DMSBatchScorer:
    def __init__(self, model, tokenizer=None, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.device = device

        if tokenizer is None:
            logger.warning("No tokenizer provided, using default ESM2...")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        else:
            self.tokenizer = tokenizer

        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = 32

    def score_mutations(self, base_data, mutations):
        scores = []
        for i in tqdm(range(0, len(mutations), self.batch_size), desc="Scoring batches"):
            batch_muts = mutations[i: i + self.batch_size]
            batch_scores = self._score_batch(base_data, batch_muts)
            scores.extend(batch_scores)
        return scores

    def _score_batch(self, base_data, batch_muts):
        B = len(batch_muts)

        # L: 输入长度 (含 CLS, EOS)，例如 288
        L = base_data.input_ids.shape[0]

        # N: 图节点数量 (真实残基数)，例如 286
        # 我们用这个 N 来计算 logits 的偏移量
        N = base_data.num_nodes

        # 1. 准备 Mask
        mask_esm = torch.ones(L, dtype=torch.long, device=self.device)
        mask_align = torch.zeros(L, dtype=torch.long, device=self.device)
        if L > 2:
            mask_align[1:-1] = 1

        # 2. 复制输入并 Mask
        input_ids = base_data.input_ids.unsqueeze(0).expand(B, -1).clone()

        target_token_ids = []
        wt_token_ids = []
        mask_positions = []

        for i, mut in enumerate(batch_muts):
            wt_aa = mut[0]
            mt_aa = mut[-1]
            pos_str = mut[1:-1]
            try:
                pos_1based = int(pos_str)
                seq_idx = pos_1based
                input_ids[i, seq_idx] = self.mask_token_id

                mt_id = self.tokenizer.convert_tokens_to_ids(mt_aa)
                wt_id = self.tokenizer.convert_tokens_to_ids(wt_aa)

                target_token_ids.append(mt_id)
                wt_token_ids.append(wt_id)

                # 这里记录的是 input_ids 里的索引 (包含 CLS)
                mask_positions.append(seq_idx)
            except:
                target_token_ids.append(0)
                wt_token_ids.append(0)
                mask_positions.append(0)

        # 3. 构造 PyG Batch List
        batch_list = []
        for i in range(B):
            new_data = base_data.clone()
            new_data.input_ids = input_ids[i]
            new_data.esm_attention_mask = mask_esm.clone()
            new_data.attention_mask = mask_align.clone()
            batch_list.append(new_data)

        batch = Batch.from_data_list(batch_list).to(self.device)

        # 4. 维度重塑
        batch.input_ids = batch.input_ids.view(B, -1)
        if hasattr(batch, 'esm_attention_mask'):
            batch.esm_attention_mask = batch.esm_attention_mask.view(B, -1)
        if hasattr(batch, 'attention_mask'):
            batch.attention_mask = batch.attention_mask.view(B, -1)

        # 5. 推理
        # logits shape: [B * N, Vocab] (例如 32 * 286 = 9152)
        with torch.no_grad():
            logits = self.model(batch)

            # 6. 提取分数 [关键修复]
        scores = []
        for i in range(B):
            # 原始 mask_pos 是针对 input_ids 的 (index 0 是 CLS, index 1 是第1个残基)
            # Logits 是针对图节点的 (index 0 是第1个残基)
            # 所以相对索引要 -1
            relative_pos = mask_positions[i] - 1

            # 绝对索引 = 当前 Batch index * 每个图的节点数 + 相对位置
            # 使用 N (286) 而不是 L (288)
            abs_idx = i * N + relative_pos

            # 防御性检查 (可选)
            if abs_idx >= logits.shape[0]:
                logger.error(f"Index Error: i={i}, N={N}, rel={relative_pos}, abs={abs_idx}, logits={logits.shape}")
                scores.append(0.0)
                continue

            mt_id = target_token_ids[i]
            wt_id = wt_token_ids[i]

            log_probs = torch.log_softmax(logits[abs_idx], dim=0)
            score = log_probs[mt_id].item() - log_probs[wt_id].item()
            scores.append(score)

        return scores