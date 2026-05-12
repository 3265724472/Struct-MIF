import os
import sys
import glob
import argparse
import torch
import pandas as pd
import logging
import re
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目根目录到 Path
sys.path.append(os.getcwd())

try:
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder
    from src.scoring import DMSBatchScorer
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder
    from src.scoring import DMSBatchScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DTm/DDG Benchmark (Multi-Epoch Sweep)")
    parser.add_argument("--dtm_root", type=str, required=True, help="数据集根目录 (e.g., data/DDG/DATASET)")
    # 🔴 [MODIFIED] 更改为输入文件夹目录
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="存放 checkpoints 的文件夹 (e.g., experiments/test2)")
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output_csv", type=str, default="dtm_sweep_results.csv")
    parser.add_argument("--foldseek_bin", type=str, default="./data/bin/foldseek")

    parser.add_argument("--mut_col", type=str, default="mutant")
    parser.add_argument("--score_col", type=str, default="score")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gvp_layers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_3di"])
    parser.add_argument("--gnn_type", type=str, default="gvp", choices=["gvp", "gcn", "gat", "egnn"])

    return parser.parse_args()


def preprocess_graph(data, tokenizer):
    if hasattr(data, 'seq_str') and data.seq_str:
        seq = data.seq_str
    elif hasattr(data, 'seq') and data.seq:
        seq = data.seq
    else:
        return None

    token_out = tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    data.input_ids = token_out["input_ids"].squeeze(0)
    mask = token_out["attention_mask"].squeeze(0)
    data.esm_attention_mask = mask
    data.attention_mask = mask.clone()

    align_mask = torch.zeros_like(data.input_ids, dtype=torch.bool)
    if len(align_mask) > 2: align_mask[1:-1] = True
    data.graph_align_mask = align_mask

    if not hasattr(data, 'x_3di'):
        if hasattr(data, 'seq_3di') and data.seq_3di is not None:
            vocab_3di = "ACDEFGHIKLMNPQRSTVWY"
            char_to_int_3di = {c: i for i, c in enumerate(vocab_3di)}
            indices = [char_to_int_3di.get(c, 0) for c in str(data.seq_3di)]
            data.x_3di = torch.tensor(indices, dtype=torch.long)
        else:
            data.x_3di = torch.zeros(len(seq), dtype=torch.long)

    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        row, col = data.edge_index
        vec = data.pos[col] - data.pos[row]
        data.edge_vectors = vec.unsqueeze(1)
    else:
        num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        data.edge_vectors = torch.zeros(num_edges, 1, 3)

    if hasattr(data, 'node_vectors'):
        n_vec = data.node_vectors[:, 0, :]
        c_vec = data.node_vectors[:, 1, :]
        cross_vec = torch.cross(n_vec, c_vec, dim=1)
        data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    return data


def auto_detect_columns(df, default_mut, default_score):
    cols = [c.lower() for c in df.columns]
    mut_col, score_col = default_mut, default_score
    possible_mut = ['mutant', 'mutation', 'variant', 'mut', 'mt']
    possible_score = ['score', 'ddg', 'dtm', 'fitness', 'dg', 'tm']

    if default_mut not in df.columns:
        for p in possible_mut:
            for c in df.columns:
                if p in c.lower(): mut_col = c; break
            if mut_col != default_mut: break
    if default_score not in df.columns:
        for p in possible_score:
            for c in df.columns:
                if p in c.lower(): score_col = c; break
            if score_col != default_score: break
    return mut_col, score_col


def extract_epoch(filename):
    """提取文件名中的 epoch 数字进行排序"""
    match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename)
    return int(match.group(1)) if match else -1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    builder = GraphBuilder(foldseek_bin_path=args.foldseek_bin)

    # 🔴 1. 扫描并构建缓存机制 (极大加速多Epoch的遍历)
    logger.info(f"🔍 Pre-loading and caching graphs from: {args.dtm_root}")
    subdirs = sorted(glob.glob(os.path.join(args.dtm_root, "*")))
    dataset_cache = []

    for subdir in tqdm(subdirs, desc="Caching Datasets"):
        if not os.path.isdir(subdir): continue
        name = os.path.basename(subdir)
        pdb_path = os.path.join(subdir, f"{name}.pdb")
        tsv_path = os.path.join(subdir, f"{name}.tsv")

        if not os.path.exists(pdb_path):
            if os.path.exists(os.path.join(subdir, f"{name}.ef.pdb")):
                pdb_path = os.path.join(subdir, f"{name}.ef.pdb")
            elif os.path.exists(os.path.join(subdir, f"{name}.exp.pdb")):
                pdb_path = os.path.join(subdir, f"{name}.exp.pdb")
            else:
                continue
        if not os.path.exists(tsv_path): continue

        try:
            base_data = builder.process(pdb_path)
            base_data = preprocess_graph(base_data, tokenizer)
            df = pd.read_csv(tsv_path, sep='\t')
            mut_col, score_col = auto_detect_columns(df, args.mut_col, args.score_col)
            if mut_col not in df.columns or score_col not in df.columns or base_data is None: continue

            mutations = df[mut_col].astype(str).tolist()
            labels = df[score_col].tolist()
            dataset_cache.append({"name": name, "data": base_data, "mutations": mutations, "labels": labels})
        except Exception as e:
            continue

    logger.info(f"✅ Successfully cached {len(dataset_cache)} valid datasets.")

    # 🔴 2. 获取所有的 Checkpoints 并排序
    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    ckpt_files.sort(key=extract_epoch)
    if not ckpt_files:
        logger.error(f"No checkpoint_epoch_X.pt files found in {args.ckpt_dir}")
        return

    # 🔴 3. 初始化模型结构 (只执行一次)
    logger.info(f"📦 Initializing Model Architecture...")
    model = StructMIF(
        esm_model_name=args.esm_model, gvp_layers=args.gvp_layers,
        gvp_node_out_dim=args.hidden_dim, top_k=args.top_k,
        ablation_mode=args.ablation, gnn_type=args.gnn_type
    )
    model.to(device)

    # 🔴 4. 遍历每个 Epoch
    all_results = []

    for ckpt_file in ckpt_files:
        epoch_idx = extract_epoch(ckpt_file)
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
        logger.info(f"==========================================")
        logger.info(f"🚀 Evaluating Epoch: {epoch_idx} | File: {ckpt_file}")

        # 热加载权重
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_sd = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if name.startswith("fusion_norm"):
                name = name.replace("fusion_norm", "fusion_module.norm")
            elif name.startswith("fusion_proj"):
                name = name.replace("fusion_proj", "fusion_module.proj")
            if name.startswith("gvp_encoder."): name = name.replace("gvp_encoder.", "encoder.")
            new_sd[name] = v

        model.load_state_dict(new_sd, strict=False)
        model.eval()
        scorer = DMSBatchScorer(model, tokenizer=tokenizer, batch_size=args.batch_size, device=device)

        epoch_results = []
        for item in tqdm(dataset_cache, desc=f"Epoch {epoch_idx}", leave=False):
            try:
                preds = scorer.score_mutations(item["data"], item["mutations"])

                # 清洗无效预测
                clean_preds, clean_labels = [], []
                for p, l in zip(preds, item["labels"]):
                    if p is not None and not pd.isna(p) and not pd.isna(l):
                        clean_preds.append(p)
                        clean_labels.append(l)

                if len(clean_preds) >= 5:
                    spr, _ = spearmanr(clean_preds, clean_labels)
                    epoch_results.append(spr)
                    all_results.append({"epoch": epoch_idx, "dataset": item["name"], "spearman": spr})
            except Exception as e:
                continue

        if epoch_results:
            avg_spr = sum(epoch_results) / len(epoch_results)
            logger.info(f"🎯 Epoch {epoch_idx} Average Spearman: {avg_spr:.4f}")

            # 实时保存，防止中断丢失
            res_df = pd.DataFrame(all_results)
            res_df.to_csv(args.output_csv, index=False)

    logger.info(f"✅ All epochs evaluated. Results securely saved to {args.output_csv}")


if __name__ == "__main__":
    main()