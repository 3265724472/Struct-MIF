import os
import sys
import argparse
import json
import torch
import pandas as pd
import logging
import re
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch

sys.path.append(os.getcwd())

try:
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class DMSBatchScorer:
    def __init__(self, model, tokenizer, batch_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def score_mutations(self, data, mutations, dataset_id="Unknown"):
        seq_str = data.seq_str if hasattr(data, 'seq_str') else data.seq
        seq_len = len(seq_str)
        preds = []

        batch = Batch.from_data_list([data]).to(self.device)
        for attr in ['input_ids', 'esm_attention_mask', 'attention_mask', 'graph_align_mask']:
            if hasattr(batch, attr) and getattr(batch, attr) is not None and getattr(batch, attr).ndim == 1:
                setattr(batch, attr, getattr(batch, attr).reshape(1, -1))

        with torch.no_grad():
            wt_logits = self.model(batch)
            wt_logits = torch.log_softmax(wt_logits, dim=-1)

        is_graph_output = (wt_logits.ndim == 2)
        offset = 1 if (is_graph_output and wt_logits.shape[0] == seq_len + 2) or (
                    not is_graph_output and wt_logits.shape[1] == seq_len + 2) else 0

        for mut_str in mutations:
            try:
                sub_parts = mut_str.split(":") if ":" in mut_str else (
                    mut_str.split(",") if "," in mut_str else [mut_str])
                valid_mut, parsed_muts = True, []

                for p in sub_parts:
                    p = p.strip()
                    if not p: continue
                    wt, pos_str, mt = p[0], p[1:-1], p[-1]
                    pos = int(pos_str) - 1
                    if pos < 0 or pos >= seq_len or seq_str[pos] != wt:
                        valid_mut = False;
                        break
                    parsed_muts.append((pos, wt, mt))

                if not valid_mut: preds.append(None); continue

                score = 0.0
                for (pos, wt, mt) in parsed_muts:
                    wt_id = self.tokenizer.convert_tokens_to_ids(wt)
                    mt_id = self.tokenizer.convert_tokens_to_ids(mt)
                    idx = pos + offset
                    val = (wt_logits[idx, mt_id] - wt_logits[idx, wt_id]).item() if is_graph_output else (
                                wt_logits[0, idx, mt_id] - wt_logits[0, idx, wt_id]).item()
                    score += val
                preds.append(score)
            except Exception:
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
    except:
        return None


def preprocess_graph(data, tokenizer, pdb_path=None):
    seq = data.seq_str if hasattr(data, 'seq_str') else getattr(data, 'seq', None)
    if not seq and pdb_path and os.path.exists(pdb_path):
        seq = get_seq_from_pdb(pdb_path);
        data.seq_str = seq;
        data.seq = seq
    if not seq: return None

    token_out = tokenizer(seq, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    data.input_ids = token_out["input_ids"].squeeze(0)
    mask = token_out["attention_mask"].squeeze(0)
    data.esm_attention_mask = mask

    align_mask = mask.clone();
    align_mask[0] = 0;
    align_mask[-1] = 0
    data.attention_mask = align_mask;
    data.graph_align_mask = align_mask.clone()

    if not hasattr(data, 'x_3di'):
        if hasattr(data, 'seq_3di') and data.seq_3di:
            mapper = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
            data.x_3di = torch.tensor([mapper.get(c, 0) for c in str(data.seq_3di)], dtype=torch.long)
        else:
            data.x_3di = torch.zeros(len(seq), dtype=torch.long)

    if hasattr(data, 'node_vectors') and data.node_vectors.shape[1] == 2:
        cross_vec = torch.cross(data.node_vectors[:, 0, :], data.node_vectors[:, 1, :], dim=1)
        data.node_vectors = torch.cat([data.node_vectors, cross_vec.unsqueeze(1)], dim=1)

    if hasattr(data, 'edge_index') and hasattr(data, 'pos'):
        try:
            row, col = data.edge_index
            data.edge_vectors = (data.pos[col] - data.pos[row]).unsqueeze(1)
        except:
            data.edge_vectors = torch.zeros(data.edge_index.shape[1] if len(data.edge_index.shape) > 1 else 0, 1, 3)
    else:
        data.edge_vectors = torch.zeros(data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0, 1, 3)
    return data


def extract_epoch(filename):
    match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename)
    return int(match.group(1)) if match else -1


def main():
    parser = argparse.ArgumentParser(description="Run ProteinGym Benchmark (Multi-Epoch Sweep)")
    parser.add_argument("--json_file", type=str, required=True)
    # 🔴 [MODIFIED] 更改为输入文件夹目录
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="存放 checkpoints 的文件夹 (e.g., experiments/test2)")
    parser.add_argument("--output_csv", type=str, default="proteingym_sweep_results.csv")
    parser.add_argument("--esm_model", type=str, default="./pretrained_models/esm2_t33_650M_UR50D")
    parser.add_argument("--foldseek_bin", type=str, default="data/bin/foldseek")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--category", type=str, default="all")

    parser.add_argument("--gvp_layers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_3di"])
    parser.add_argument("--gnn_type", type=str, default="gvp", choices=["gvp", "gcn", "gat", "egnn"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    graph_builder = GraphBuilder(foldseek_bin_path=args.foldseek_bin)

    with open(args.json_file, 'r') as f:
        json_data = json.load(f)
    all_tasks = json_data if isinstance(json_data, list) else [item | {'category': cat} for cat, items in
                                                               json_data.items() for item in items]
    target_category = args.category.lower().strip()
    tasks = all_tasks if target_category == "all" else [t for t in all_tasks if
                                                        target_category in t.get('category', '').lower()]

    # 🔴 1. 扫描并构建缓存机制
    logger.info(f"🔍 Pre-loading and caching {len(tasks)} ProteinGym graphs...")
    dataset_cache = []
    for task in tqdm(tasks, desc="Caching Datasets"):
        dms_path, pdb_path = task.get('dms_path'), task.get('pdb_path')
        if not os.path.exists(pdb_path) or not os.path.exists(dms_path): continue
        try:
            df = pd.read_csv(dms_path)
            mut_col = next((c for c in df.columns if 'mut' in c.lower()), None)
            score_col = next((c for c in df.columns if 'score' in c.lower()), None)
            if not mut_col or not score_col: continue

            base_data = graph_builder.process(pdb_path)
            base_data = preprocess_graph(base_data, tokenizer, pdb_path)
            if base_data is None: continue

            dataset_cache.append({
                "id": task.get('id', 'unknown'),
                "category": task.get('category', 'Uncategorized'),
                "data": base_data,
                "mutations": df[mut_col].tolist(),
                "labels": df[score_col].tolist()
            })
        except Exception as e:
            continue

    logger.info(f"✅ Successfully cached {len(dataset_cache)} valid ProteinGym tasks.")

    # 🔴 2. 获取所有的 Checkpoints 并排序
    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    ckpt_files.sort(key=extract_epoch)
    if not ckpt_files: return logger.error("No valid checkpoints found!")

    # 🔴 3. 初始化模型
    model = StructMIF(esm_model_name=args.esm_model, gvp_layers=args.gvp_layers, gvp_node_out_dim=args.hidden_dim,
                      top_k=args.top_k, ablation_mode=args.ablation, gnn_type=args.gnn_type).to(device)

    # 🔴 4. 遍历每个 Epoch
    all_results = []
    for ckpt_file in ckpt_files:
        epoch_idx = extract_epoch(ckpt_file)
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
        logger.info(f"==========================================")
        logger.info(f"🚀 Evaluating Epoch: {epoch_idx} | File: {ckpt_file}")

        # 热加载权重
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict.get('model_state_dict', state_dict)
        new_sd = {k.replace('module.', '').replace("fusion_norm", "fusion_module.norm").replace("fusion_proj",
                                                                                                "fusion_module.proj").replace(
            "gvp_encoder.", "encoder."): v for k, v in state_dict.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        scorer = DMSBatchScorer(model, tokenizer, args.batch_size, device)

        epoch_rhos = []
        for item in tqdm(dataset_cache, desc=f"Epoch {epoch_idx}", leave=False):
            try:
                preds = scorer.score_mutations(item["data"], item["mutations"])
                clean_preds, clean_labels = zip(*[(p, l) for p, l in zip(preds, item["labels"]) if
                                                  p is not None and not pd.isna(p) and not pd.isna(l)])
                if len(clean_preds) > 5:
                    sp_res = spearmanr(clean_preds, clean_labels)
                    rho = sp_res[0] if isinstance(sp_res, tuple) else (
                        sp_res.statistic if hasattr(sp_res, 'statistic') else sp_res)
                    epoch_rhos.append(rho)
                    all_results.append(
                        {"epoch": epoch_idx, "id": item["id"], "category": item["category"], "spearman": rho,
                         "N": len(clean_preds)})
            except Exception:
                continue

        if epoch_rhos:
            avg_rho = sum(epoch_rhos) / len(epoch_rhos)
            logger.info(f"🎯 Epoch {epoch_idx} Overall ProteinGym Spearman: {avg_rho:.4f}")
            pd.DataFrame(all_results).to_csv(args.output_csv, index=False)

    logger.info(f"✅ All {len(ckpt_files)} epochs evaluated. Final table saved to {args.output_csv}")


if __name__ == "__main__":
    main()