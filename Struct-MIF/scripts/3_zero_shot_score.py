import os
import sys
import argparse
import torch
import pandas as pd
import logging
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目根目录到 Path
sys.path.append(os.getcwd())

from src.modeling.struct_mif import StructMIF
from src.data.graph_builder import GraphBuilder
from src.scoring import DMSBatchScorer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Struct-MIF Zero-Shot 突变打分脚本")
    parser.add_argument("--pdb_file", type=str, required=True, help="野生型蛋白质 PDB 路径")
    parser.add_argument("--csv_file", type=str, required=True, help="突变列表 CSV 文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="预训练模型权重 (.pt)")
    parser.add_argument("--mut_col", type=str, default="mutation")
    parser.add_argument("--score_col", type=str, default="score")
    parser.add_argument("--foldseek_bin", type=str, default="./data/bin/foldseek")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gvp_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def preprocess_graph(data, tokenizer):
    """
    预处理图数据
    [修复] 添加 graph_align_mask 以解决 9216 != 9152 的维度不匹配问题
    """
    # 1. Input IDs & Masks
    token_out = tokenizer(data.seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)

    data.input_ids = token_out["input_ids"].squeeze(0)

    # Attention Mask (用于 ESM 内部处理 Padding)
    mask = token_out["attention_mask"].squeeze(0)
    data.esm_attention_mask = mask
    data.attention_mask = mask.clone()

    # [关键修复] Graph Align Mask (用于过滤掉 CLS 和 EOS)
    # ESM 输出: [CLS, A, A, ..., A, EOS]
    # 我们只需要中间的 A...A 对应图节点
    align_mask = torch.zeros_like(data.input_ids, dtype=torch.bool)
    # 设中间部分为 True (去除首尾)
    if len(align_mask) > 2:
        align_mask[1:-1] = True
    data.graph_align_mask = align_mask

    # 2. x_3di
    vocab_3di = "ACDEFGHIKLMNPQRSTVWY"
    char_to_int_3di = {c: i for i, c in enumerate(vocab_3di)}
    if hasattr(data, 'seq_3di') and data.seq_3di is not None:
        indices = [char_to_int_3di.get(c, 0) for c in data.seq_3di]
        data.x_3di = torch.tensor(indices, dtype=torch.long)
    else:
        data.x_3di = torch.zeros(data.num_nodes, dtype=torch.long)

    # 3. Edge Vectors
    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        row, col = data.edge_index
        vec = data.pos[col] - data.pos[row]
        data.edge_vectors = vec.unsqueeze(1)
    else:
        num_edges = data.edge_index.shape[1]
        data.edge_vectors = torch.zeros(num_edges, 1, 3)

    # 4. Node Vectors
    if hasattr(data, 'node_vectors'):
        n_vec = data.node_vectors[:, 0, :]
        c_vec = data.node_vectors[:, 1, :]
        cross_vec = torch.cross(n_vec, c_vec, dim=1)
        data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    return data


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    logger.info(f"Loading Struct-MIF model from {args.checkpoint}...")
    model = StructMIF(esm_model_name=args.esm_model, gvp_layers=args.gvp_layers)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer from {args.esm_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)

    # 3. Process PDB
    logger.info(f"Processing PDB: {args.pdb_file}...")
    builder = GraphBuilder(foldseek_bin_path=args.foldseek_bin)
    base_data = builder.process(args.pdb_file)
    if base_data is None: return

    logger.info("Preprocessing graph data...")
    base_data = preprocess_graph(base_data, tokenizer)

    # 4. Scoring
    df = pd.read_csv(args.csv_file)
    mutations = df[args.mut_col].tolist()
    logger.info(f"Found {len(mutations)} mutations.")

    logger.info("Starting Zero-shot inference...")
    scorer = DMSBatchScorer(model, tokenizer=tokenizer, batch_size=args.batch_size, device=device)

    try:
        pred_scores = scorer.score_mutations(base_data, mutations)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Save
    df['struct_mif_score'] = pred_scores
    if args.score_col in df.columns:
        valid_df = df.dropna(subset=[args.score_col, 'struct_mif_score'])
        if len(valid_df) > 0:
            spr, p_val = spearmanr(valid_df[args.score_col], valid_df['struct_mif_score'])
            logger.info(f"📊 Spearman: {spr:.4f} (p={p_val:.4e})")

    output_path = args.output_csv if args.output_csv else args.csv_file.replace(".csv", "_pred.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()