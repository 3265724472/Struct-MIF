import os
import sys
import argparse
import json
import torch
import pandas as pd
import logging
import numpy as np
import traceback
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch

# 添加项目根目录
sys.path.append(os.getcwd())

# 尝试导入核心模块
try:
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    try:
        from src.modeling.struct_mif import StructMIF
        from src.data.graph_builder import GraphBuilder
    except ImportError:
        pass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ==========================================
# 🔥 完整版 Scorer (支持多点突变 + 维度自适应)
# ==========================================
class DMSBatchScorer:
    def __init__(self, model, tokenizer, batch_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def score_mutations(self, data, mutations, dataset_id="Unknown"):
        # 1. 获取序列
        if hasattr(data, 'seq_str') and data.seq_str:
            seq_str = data.seq_str
        elif hasattr(data, 'seq') and data.seq:
            seq_str = data.seq
        else:
            return [None] * len(mutations)

        seq_len = len(seq_str)
        preds = []

        # 2. 创建单样本 Batch
        batch = Batch.from_data_list([data]).to(self.device)

        if batch.input_ids.ndim == 1:
            batch.input_ids = batch.input_ids.reshape(1, -1)
        if hasattr(batch, 'esm_attention_mask') and batch.esm_attention_mask is not None:
            if batch.esm_attention_mask.ndim == 1:
                batch.esm_attention_mask = batch.esm_attention_mask.reshape(1, -1)
        if hasattr(batch, 'attention_mask') and batch.attention_mask is not None:
            if batch.attention_mask.ndim == 1:
                batch.attention_mask = batch.attention_mask.reshape(1, -1)
        if hasattr(batch, 'graph_align_mask') and batch.graph_align_mask is not None:
            if batch.graph_align_mask.ndim == 1:
                batch.graph_align_mask = batch.graph_align_mask.reshape(1, -1)

        # 3. 预计算野生型 Logits
        with torch.no_grad():
            wt_logits = self.model(batch)  # [Nodes, 33]
            wt_logits = torch.log_softmax(wt_logits, dim=-1)

        is_graph_output = (wt_logits.ndim == 2)

        if is_graph_output:
            offset = 0
            if wt_logits.shape[0] == seq_len + 2:
                offset = 1
        else:
            offset = 1 if wt_logits.shape[1] == seq_len + 2 else 0

        # 4. 逐个处理突变
        for mut_str in tqdm(mutations, desc=f"Scoring {dataset_id}", leave=False):
            try:
                if ":" in mut_str:
                    sub_parts = mut_str.split(":")
                elif "," in mut_str:
                    sub_parts = mut_str.split(",")
                else:
                    sub_parts = [mut_str]

                valid_mut = True
                parsed_muts = []

                for p in sub_parts:
                    p = p.strip()
                    if not p: continue
                    try:
                        wt, pos_str, mt = p[0], p[1:-1], p[-1]
                        pos = int(pos_str) - 1
                    except:
                        valid_mut = False;
                        break

                    if pos < 0 or pos >= seq_len:
                        valid_mut = False;
                        break

                    if seq_str[pos] != wt:
                        valid_mut = False;
                        break

                    parsed_muts.append((pos, wt, mt))

                if not valid_mut or not parsed_muts:
                    preds.append(None);
                    continue

                score = 0.0
                for (pos, wt, mt) in parsed_muts:
                    wt_id = self.tokenizer.convert_tokens_to_ids(wt)
                    mt_id = self.tokenizer.convert_tokens_to_ids(mt)

                    idx = pos + offset
                    if is_graph_output:
                        val = (wt_logits[idx, mt_id] - wt_logits[idx, wt_id]).item()
                    else:
                        val = (wt_logits[0, idx, mt_id] - wt_logits[0, idx, wt_id]).item()

                    score += val

                preds.append(score)

            except Exception as e:
                preds.append(None)

        return preds


def get_seq_from_pdb(pdb_path):
    try:
        three_to_one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
        seq = []
        with open(pdb_path, 'r') as f:
            last_res_id = None
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    res_id = line[22:27].strip()
                    if res_id != last_res_id:
                        seq.append(three_to_one.get(res_name, 'X'))
                        last_res_id = res_id
        return "".join(seq)
    except Exception as e:
        return None


def preprocess_graph(data, tokenizer, pdb_path=None):
    seq = None
    if hasattr(data, 'seq_str') and data.seq_str:
        seq = data.seq_str
    elif hasattr(data, 'seq') and data.seq:
        seq = data.seq

    if not seq and pdb_path and os.path.exists(pdb_path):
        seq = get_seq_from_pdb(pdb_path)
        data.seq_str = seq
        data.seq = seq

    if not seq: return None

    token_out = tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    data.input_ids = token_out["input_ids"].squeeze(0)
    mask = token_out["attention_mask"].squeeze(0)
    data.esm_attention_mask = mask

    align_mask = mask.clone()
    align_mask[0] = 0
    align_mask[-1] = 0
    data.attention_mask = align_mask
    data.graph_align_mask = align_mask.clone()

    if not hasattr(data, 'x_3di'):
        if hasattr(data, 'seq_3di') and data.seq_3di:
            vocab = "ACDEFGHIKLMNPQRSTVWY"
            mapper = {c: i for i, c in enumerate(vocab)}
            indices = [mapper.get(c, 0) for c in str(data.seq_3di)]
            data.x_3di = torch.tensor(indices, dtype=torch.long)
        else:
            data.x_3di = torch.zeros(len(seq), dtype=torch.long)

    if hasattr(data, 'node_vectors'):
        if data.node_vectors.shape[1] == 2:
            n_vec = data.node_vectors[:, 0, :]
            c_vec = data.node_vectors[:, 1, :]
            cross_vec = torch.cross(n_vec, c_vec, dim=1)
            data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        try:
            row, col = data.edge_index
            vec = data.pos[col] - data.pos[row]
            data.edge_vectors = vec.unsqueeze(1)
        except:
            num_edges = data.edge_index.shape[1] if len(data.edge_index.shape) > 1 else 0
            data.edge_vectors = torch.zeros(num_edges, 1, 3)
    else:
        num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        data.edge_vectors = torch.zeros(num_edges, 1, 3)

    return data


def main():
    parser = argparse.ArgumentParser(description="Run ProteinGym Benchmark")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="proteingym_results.csv")
    parser.add_argument("--esm_model", type=str, default="./pretrained_models/esm2_t33_650M_UR50D")
    parser.add_argument("--foldseek_bin", type=str, default="data/bin/foldseek")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--category", type=str, default="all")

    # 🔴 [实验参数] 模型结构控制
    parser.add_argument("--gvp_layers", type=int, default=6, help="Depth: Number of GNN layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Width: Hidden dimension size")
    parser.add_argument("--top_k", type=int, default=30, help="Topology: Number of KNN neighbors")

    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "no_3di"],
                        help="Ablation mode used during training")
    parser.add_argument("--gnn_type", type=str, default="gvp",
                        choices=["gvp", "gcn", "gat", "egnn"],
                        help="Choose GNN encoder: gvp (geometric), egnn (distance), gcn/gat (topology only)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"📦 Loading Model (Ablation: {args.ablation})...")
    logger.info(f"⚙️ Config: GNN={args.gnn_type} | K={args.top_k} | Dim={args.hidden_dim} | Layers={args.gvp_layers}")

    try:
        model = StructMIF(
            esm_model_name=args.esm_model,
            gvp_layers=args.gvp_layers,
            gvp_node_out_dim=args.hidden_dim,  # 🔴 传入隐层维度
            top_k=args.top_k,  # 🔴 传入 K 值
            ablation_mode=args.ablation,
            gnn_type=args.gnn_type
        )
    except NameError:
        logger.error("❌ Failed to import StructMIF.")
        return

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

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
    except:
        logger.warning("⚠️ Strict load failed, trying non-strict...")
        model.load_state_dict(new_sd, strict=False)

    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    graph_builder = GraphBuilder(foldseek_bin_path=args.foldseek_bin)

    scorer = DMSBatchScorer(model, tokenizer, args.batch_size, device)

    if not os.path.exists(args.json_file):
        logger.error(f"JSON file not found: {args.json_file}")
        return

    with open(args.json_file, 'r') as f:
        json_data = json.load(f)

    all_tasks = []
    if isinstance(json_data, list):
        all_tasks = json_data
    elif isinstance(json_data, dict):
        for cat, items in json_data.items():
            for item in items:
                if 'category' not in item: item['category'] = cat
                all_tasks.append(item)

    tasks = []
    target_category = args.category.lower().strip()
    if target_category == "all":
        tasks = all_tasks
    else:
        for task in all_tasks:
            task_cat = task.get('category', '').lower()
            if target_category in task_cat:
                tasks.append(task)

    logger.info(f"📋 Loaded {len(tasks)} tasks (Category filter: {args.category}).")
    results = []

    for task in tqdm(tasks, desc=f"Benchmarking ({args.category})", leave=True):
        dataset_id = task.get('id', 'unknown')
        pdb_path = task.get('pdb_path')
        dms_path = task.get('dms_path')
        category = task.get('category', 'Uncategorized')

        if not os.path.exists(pdb_path) or not os.path.exists(dms_path): continue

        try:
            if device.type == 'cuda': torch.cuda.empty_cache()

            try:
                data = graph_builder.process(pdb_path)
            except:
                data = None
            if not data: continue

            data = preprocess_graph(data, tokenizer, pdb_path)
            if not data: continue

            try:
                df = pd.read_csv(dms_path)
            except:
                continue

            mut_col = next((c for c in df.columns if 'mut' in c.lower()), None)
            score_col = next((c for c in df.columns if 'score' in c.lower()), None)

            if not mut_col or not score_col: continue

            muts = df[mut_col].tolist()
            labels = df[score_col].tolist()

            preds = scorer.score_mutations(data, muts, dataset_id)

            clean_preds, clean_labels = [], []
            for p, l in zip(preds, labels):
                if p is not None and not pd.isna(p) and not pd.isna(l):
                    clean_preds.append(p)
                    clean_labels.append(l)

            if len(clean_preds) > 5:
                sp_res = spearmanr(clean_preds, clean_labels)
                try:
                    rho = sp_res[0]
                except:
                    rho = sp_res.statistic if hasattr(sp_res, 'statistic') else sp_res

                results.append({
                    "id": dataset_id,
                    "category": category,
                    "spearman": rho,
                    "N": len(clean_preds)
                })

        except Exception as e:
            logger.error(f"Error on {dataset_id}: {e}")

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(args.output_csv, index=False)

        print("\n" + "=" * 40)
        print(f"🏆 ProteinGym Results ({args.category})")
        print("=" * 40)
        summary = res_df.groupby("category")["spearman"].agg(['mean', 'count', 'std'])
        summary.columns = ['Avg Spearman', 'Count', 'Std']
        print(summary)
        print("-" * 40)
        print(f"Overall Average: {res_df['spearman'].mean():.4f}")
        print("=" * 40)
    else:
        logger.error("No results generated!")


if __name__ == "__main__":
    main()