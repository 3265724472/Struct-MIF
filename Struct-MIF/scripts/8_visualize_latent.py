import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics import silhouette_score

# 添加项目根目录到 Path
sys.path.append(os.getcwd())

try:
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from src.modeling.struct_mif import StructMIF
    from src.data.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_graph(data, tokenizer):
    """图数据与特殊 Token 的维度对齐预处理"""
    seq = data.seq_str if hasattr(data, 'seq_str') else getattr(data, 'seq', None)
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


def main():
    parser = argparse.ArgumentParser(description="Extract Latent Features and Plot UMAP using JSON")
    parser.add_argument("--json_file", type=str, required=True, help="protssn_experiment_plan.json 的路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型权重 (推荐用最佳 epoch)")
    parser.add_argument("--esm_model", type=str, default="./pretrained_models/esm2_t33_650M_UR50D")
    parser.add_argument("--foldseek_bin", type=str, default="data/bin/foldseek")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 1. 读取 JSON 任务文件 =================
    logger.info(f"📂 Loading JSON task file: {args.json_file}")
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)

    records = []
    for category, items in json_data.items():
        for item in items:
            records.append({
                'pdb_path': item['pdb_path'],
                'class_name': category
            })
    df = pd.DataFrame(records)
    logger.info(f"✅ Loaded {len(df)} proteins across {df['class_name'].nunique()} functional categories.")

    # ================= 2. 加载模型与 Hook =================
    logger.info("📦 Loading Model Architecture...")
    model = StructMIF(esm_model_name=args.esm_model, gvp_layers=6, gvp_node_out_dim=128, top_k=30)

    logger.info(f"🔄 Loading Checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state_dict.get('model_state_dict', state_dict)
    new_sd = {k.replace('module.', '').replace("fusion_norm", "fusion_module.norm").replace("fusion_proj",
                                                                                            "fusion_module.proj").replace(
        "gvp_encoder.", "encoder."): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()

    extracted_features = {'esm': [], 'gvp': []}

    def esm_hook(module, input, output):
        hidden_states = output.last_hidden_state if hasattr(output, 'last_hidden_state') else (
            output[0] if isinstance(output, tuple) else output)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)  # [Seq_len, 1280]
        extracted_features['esm'] = hidden_states

    def gvp_hook(module, input, output):
        s_out = output[0] if isinstance(output, tuple) else output
        extracted_features['gvp'] = s_out  # [Seq_len, 128]

    hook_esm = model.esm_encoder.register_forward_hook(esm_hook)
    hook_gvp = model.encoder.register_forward_hook(gvp_hook)

    # ================= 3. 提取特征 =================
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    builder = GraphBuilder(foldseek_bin_path=args.foldseek_bin)

    all_esm_pools = []
    all_gvp_pools = []
    all_labels = []

    logger.info("🧬 Processing PDBs and Extracting Global Graph Features...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdb_path = row['pdb_path']
        cls_name = row['class_name']

        if not os.path.exists(pdb_path):
            continue

        try:
            data = builder.process(pdb_path)
            data = preprocess_graph(data, tokenizer)
            if data is None: continue

            batch = Batch.from_data_list([data]).to(device)
            for attr in ['input_ids', 'esm_attention_mask', 'attention_mask', 'graph_align_mask']:
                if hasattr(batch, attr) and getattr(batch, attr) is not None and getattr(batch, attr).ndim == 1:
                    setattr(batch, attr, getattr(batch, attr).reshape(1, -1))

            with torch.no_grad():
                _ = model(batch)

            esm_nodes = extracted_features['esm']
            gvp_nodes = extracted_features['gvp']

            num_nodes = gvp_nodes.shape[0]

            if esm_nodes.shape[0] == num_nodes + 2:
                esm_nodes = esm_nodes[1:num_nodes + 1]

            esm_graph_feat = global_mean_pool(esm_nodes, batch.batch).cpu().numpy()
            gvp_graph_feat = global_mean_pool(gvp_nodes, batch.batch).cpu().numpy()

            all_esm_pools.append(esm_graph_feat)
            all_gvp_pools.append(gvp_graph_feat)
            all_labels.append(cls_name)

        except Exception as e:
            continue

    hook_esm.remove()
    hook_gvp.remove()

    if not all_esm_pools:
        logger.error("❌ No features extracted! Please check your PDB paths in JSON.")
        return

    # ================= 4. 使用 UMAP 降维与并排绘图 =================
    X_esm = np.concatenate(all_esm_pools, axis=0)  # [N, 1280]
    X_gvp = np.concatenate(all_gvp_pools, axis=0)  # [N, 128]

    logger.info(f"🎨 Running UMAP dimensionality reduction on {len(all_labels)} proteins...")

    # 4.1 执行 UMAP 降维
    reducer_esm = UMAP(n_neighbors=15, min_dist=0.2, metric='cosine', random_state=42)
    emb_esm = reducer_esm.fit_transform(X_esm)

    reducer_gvp = UMAP(n_neighbors=15, min_dist=0.2, metric='cosine', random_state=42)
    emb_gvp = reducer_gvp.fit_transform(X_gvp)

    # 4.2 计算 Silhouette Score
    sil_esm = silhouette_score(emb_esm, all_labels)
    sil_gvp = silhouette_score(emb_gvp, all_labels)

    logger.info(f"📊 ESM-2 Silhouette Score: {sil_esm:.4f}")
    logger.info(f"📊 Struct-MIF Silhouette Score: {sil_gvp:.4f}")

    # 4.3 构建一幅包含左右两列的画板
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # 1行2列，宽度加宽到18
    palette = sns.color_palette("Set2", len(set(all_labels)))

    df_esm = pd.DataFrame({'UMAP 1': emb_esm[:, 0], 'UMAP 2': emb_esm[:, 1], 'Function': all_labels})
    df_gvp = pd.DataFrame({'UMAP 1': emb_gvp[:, 0], 'UMAP 2': emb_gvp[:, 1], 'Function': all_labels})

    # 绘制左图 (ESM-2)
    sns.scatterplot(
        x='UMAP 1', y='UMAP 2', hue='Function',
        palette=palette, data=df_esm, alpha=0.85, edgecolor="w", s=80,
        ax=axes[0], legend=False  # 左图不显示图例，避免重复
    )
    axes[0].set_title(f"Frozen ESM-2 (Semantic Only)\nSilhouette Score: {sil_esm:.3f}", fontsize=15, fontweight='bold',
                      pad=15)
    axes[0].set_xlabel("UMAP 1", fontsize=12)
    axes[0].set_ylabel("UMAP 2", fontsize=12)

    # 绘制右图 (Struct-MIF)
    sns.scatterplot(
        x='UMAP 1', y='UMAP 2', hue='Function',
        palette=palette, data=df_gvp, alpha=0.85, edgecolor="w", s=80,
        ax=axes[1]
    )
    axes[1].set_title(f"Struct-MIF (Multi-modal + GVP)\nSilhouette Score: {sil_gvp:.3f}", fontsize=15,
                      fontweight='bold', pad=15)
    axes[1].set_xlabel("UMAP 1", fontsize=12)
    axes[1].set_ylabel("UMAP 2", fontsize=12)

    # 统一设置共享图例 (放在右图的右侧)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Biological Function", fontsize=12,
                   title_fontsize=13)

    plt.tight_layout()
    output_filename = "umap_combined_gym.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"💾 Saved combined plot to {output_filename}")


if __name__ == "__main__":
    main()