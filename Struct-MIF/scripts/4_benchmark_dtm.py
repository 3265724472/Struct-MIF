import os
import sys
import glob
import argparse
import torch
import pandas as pd
import logging
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目根目录到 Path
sys.path.append(os.getcwd())

# 尝试导入模块
try:
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder
    from src.scoring import DMSBatchScorer
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
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
    parser = argparse.ArgumentParser(description="DTm/DDG Benchmark")
    parser.add_argument("--dtm_root", type=str, required=True, help="数据集根目录 (e.g., data/DDG/DATASET)")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重 (.pt)")
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output_csv", type=str, default="dtm_results.csv")
    parser.add_argument("--foldseek_bin", type=str, default="./data/bin/foldseek")

    # 列名设置
    parser.add_argument("--mut_col", type=str, default="mutant")
    parser.add_argument("--score_col", type=str, default="score")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # 🔴 [实验参数] 模型结构控制
    parser.add_argument("--gvp_layers", type=int, default=6, help="Depth: Number of GNN layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Width: Hidden dimension size")
    parser.add_argument("--top_k", type=int, default=30, help="Topology: Number of KNN neighbors")

    # 消融实验参数
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "no_3di"],
                        help="Ablation mode used during training")
    parser.add_argument("--gnn_type", type=str, default="gvp",
                        choices=["gvp", "gcn", "gat", "egnn"],
                        help="Choose GNN encoder: gvp (geometric), egnn (distance), gcn/gat (topology only)")

    return parser.parse_args()


def preprocess_graph(data, tokenizer):
    """
    预处理图数据
    """
    if hasattr(data, 'seq_str') and data.seq_str:
        seq = data.seq_str
    elif hasattr(data, 'seq') and data.seq:
        seq = data.seq
    else:
        return None

    # 1. Tokenize
    token_out = tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    data.input_ids = token_out["input_ids"].squeeze(0)

    mask = token_out["attention_mask"].squeeze(0)
    data.esm_attention_mask = mask
    data.attention_mask = mask.clone()

    # Graph Align Mask
    align_mask = torch.zeros_like(data.input_ids, dtype=torch.bool)
    if len(align_mask) > 2:
        align_mask[1:-1] = True
    data.graph_align_mask = align_mask

    # 2. x_3di
    if not hasattr(data, 'x_3di'):
        if hasattr(data, 'seq_3di') and data.seq_3di is not None:
            vocab_3di = "ACDEFGHIKLMNPQRSTVWY"
            char_to_int_3di = {c: i for i, c in enumerate(vocab_3di)}
            indices = [char_to_int_3di.get(c, 0) for c in str(data.seq_3di)]
            data.x_3di = torch.tensor(indices, dtype=torch.long)
        else:
            num_nodes = len(seq)
            data.x_3di = torch.zeros(num_nodes, dtype=torch.long)

    # 3. Edge Vectors
    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        row, col = data.edge_index
        vec = data.pos[col] - data.pos[row]
        data.edge_vectors = vec.unsqueeze(1)
    else:
        num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        data.edge_vectors = torch.zeros(num_edges, 1, 3)

    # 4. Node Vectors
    if hasattr(data, 'node_vectors'):
        n_vec = data.node_vectors[:, 0, :]
        c_vec = data.node_vectors[:, 1, :]
        cross_vec = torch.cross(n_vec, c_vec, dim=1)
        data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    return data


def auto_detect_columns(df, default_mut, default_score):
    """自动检测 TSV 列名"""
    cols = [c.lower() for c in df.columns]
    mut_col = default_mut
    score_col = default_score

    possible_mut = ['mutant', 'mutation', 'variant', 'mut', 'mt']
    possible_score = ['score', 'ddg', 'dtm', 'fitness', 'dg', 'tm']

    if default_mut not in df.columns:
        for p in possible_mut:
            for c in df.columns:
                if p in c.lower():
                    mut_col = c
                    break
            if mut_col != default_mut: break

    if default_score not in df.columns:
        for p in possible_score:
            for c in df.columns:
                if p in c.lower():
                    score_col = c
                    break
            if score_col != default_score: break

    return mut_col, score_col


def process_single_dataset(subdir, model, tokenizer, builder, args, device):
    """处理单个子文件夹"""
    name = os.path.basename(subdir)
    pdb_path = os.path.join(subdir, f"{name}.pdb")
    tsv_path = os.path.join(subdir, f"{name}.tsv")

    # 检查文件是否存在
    if not os.path.exists(pdb_path):
        if os.path.exists(os.path.join(subdir, f"{name}.ef.pdb")):
            pdb_path = os.path.join(subdir, f"{name}.ef.pdb")
        elif os.path.exists(os.path.join(subdir, f"{name}.exp.pdb")):
            pdb_path = os.path.join(subdir, f"{name}.exp.pdb")
        else:
            return None

    if not os.path.exists(tsv_path):
        return None

    # 1. 解析 PDB
    try:
        base_data = builder.process(pdb_path)
    except Exception as e:
        logger.warning(f"GraphBuilder error on {name}: {e}")
        return None

    if base_data is None: return None

    # 2. 预处理
    base_data = preprocess_graph(base_data, tokenizer)
    if base_data is None: return None

    # 3. 读取 TSV
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception:
        return None

    # 自动列名匹配
    mut_col, score_col = auto_detect_columns(df, args.mut_col, args.score_col)

    if mut_col not in df.columns or score_col not in df.columns:
        return None

    mutations = df[mut_col].astype(str).tolist()

    # 4. 推理
    scorer = DMSBatchScorer(model, tokenizer=tokenizer, batch_size=args.batch_size, device=device)
    try:
        pred_scores = scorer.score_mutations(base_data, mutations)
    except Exception as e:
        logger.error(f"Inference error for {name}: {e}")
        return None

    # 5. 计算相关性
    df['pred'] = pred_scores
    valid_df = df.dropna(subset=[score_col, 'pred'])

    if len(valid_df) < 5:
        return None

    try:
        spr, p_val = spearmanr(valid_df[score_col], valid_df['pred'])
        return spr
    except:
        return None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    logger.info(f"📦 Loading Model (Ablation: {args.ablation})...")
    logger.info(f"⚙️ Config: GNN={args.gnn_type} | K={args.top_k} | Dim={args.hidden_dim} | Layers={args.gvp_layers}")

    model = StructMIF(
        esm_model_name=args.esm_model,
        gvp_layers=args.gvp_layers,
        gvp_node_out_dim=args.hidden_dim,  # 🔴 传入隐层维度
        top_k=args.top_k,  # 🔴 传入 K 值
        ablation_mode=args.ablation,
        gnn_type=args.gnn_type
    )

    logger.info(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 🔴 Key 重命名补丁
    new_sd = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        if name.startswith("fusion_norm"):
            name = name.replace("fusion_norm", "fusion_module.norm")
        elif name.startswith("fusion_proj"):
            name = name.replace("fusion_proj", "fusion_module.proj")

        # 兼容旧版的 GVP 变量名
        if name.startswith("gvp_encoder."):
            name = name.replace("gvp_encoder.", "encoder.")

        new_sd[name] = v

    try:
        model.load_state_dict(new_sd, strict=True)
        logger.info("✅ Weights loaded perfectly (Strict mode).")
    except RuntimeError as e:
        logger.warning(f"⚠️ Strict Load failed, trying loose load: {e}")
        model.load_state_dict(new_sd, strict=False)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    builder = GraphBuilder(foldseek_bin_path=args.foldseek_bin)

    # 2. 扫描数据集
    logger.info(f"Scanning subdirectories in: {args.dtm_root}")
    subdirs = sorted(glob.glob(os.path.join(args.dtm_root, "*")))
    logger.info(f"🔍 Found {len(subdirs)} potential dataset folders.")

    results = []

    for subdir in tqdm(subdirs):
        if not os.path.isdir(subdir): continue

        spr = process_single_dataset(subdir, model, tokenizer, builder, args, device)

        if spr is not None:
            name = os.path.basename(subdir)
            results.append({"dataset": name, "spearman": spr})
            logger.info(f"{name}: {spr:.4f}")

    # 3. 汇总
    if len(results) > 0:
        res_df = pd.DataFrame(results)
        res_df.to_csv(args.output_csv, index=False)
        mean_spr = res_df['spearman'].mean()
        logger.info(f"✅ Average Spearman: {mean_spr:.4f}")
        logger.info(f"Saved to {args.output_csv}")
    else:
        logger.warning("No valid results computed.")


if __name__ == "__main__":
    main()