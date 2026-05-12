import os
import sys
import torch
import logging
from torch_geometric.data import Batch

# 添加项目根目录到 Path
sys.path.append(os.getcwd())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_imports():
    """1. 检查核心库安装"""
    logger.info(">>> Step 1: Checking Imports...")
    try:
        import torch_geometric
        import gvp
        import transformers
        from src.modeling.struct_mif import StructMIF
        from src.loss import MaskedMLMLoss
        logger.info("✅ All libraries imported successfully.")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error("Please check your environment installation.")
        sys.exit(1)


def get_dummy_batch(batch_size=2, seq_len=16, device="cpu"):
    """
    构造一个符合 StructMIF 输入要求的伪造 Batch
    模拟 src.data.collator.StructMIFCollator 的输出
    """
    # 1. 序列数据 (ESM)
    # 长度 = L + 2 (CLS + EOS)
    total_len = seq_len + 2
    input_ids = torch.randint(0, 30, (batch_size, total_len)).to(device)

    # ESM Attention Mask (全 1)
    esm_attention_mask = torch.ones((batch_size, total_len), dtype=torch.long).to(device)

    # 2. 对齐 Mask (Graph Alignment Mask)
    # 中间是 1 (氨基酸)，两头是 0 (CLS/EOS)
    align_mask = torch.zeros((batch_size, total_len), dtype=torch.long).to(device)
    align_mask[:, 1:-1] = 1

    # 3. 图数据 (GVP)
    # 图节点数 = Batch * Seq_Len (不含 CLS/EOS)
    total_nodes = batch_size * seq_len

    # 3Di Token (0-20)
    x_3di = torch.randint(0, 20, (total_nodes,)).to(device)

    # 节点向量 [N, 3, 3]
    node_vectors = torch.randn(total_nodes, 3, 3).to(device)

    # 边索引 (构建一个简单的环状图)
    edge_src = torch.arange(0, total_nodes).to(device)
    edge_dst = torch.roll(edge_src, -1)
    edge_index = torch.stack([edge_src, edge_dst], dim=0).to(device)

    # 边向量 [E, 1, 3]
    num_edges = edge_index.shape[1]
    edge_vectors = torch.randn(num_edges, 1, 3).to(device)

    # PyG Batch 向量 (标示每个节点属于哪个图)
    batch_idx = []
    for i in range(batch_size):
        batch_idx.extend([i] * seq_len)
    batch_vec = torch.tensor(batch_idx, dtype=torch.long).to(device)

    # 4. Labels (用于 Loss)
    labels = input_ids.clone()

    # 封装成 PyG Batch 对象
    data = Batch(
        batch=batch_vec,
        x_3di=x_3di,
        node_vectors=node_vectors,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        num_nodes=total_nodes
    )

    # 挂载序列相关属性
    data.input_ids = input_ids
    data.esm_attention_mask = esm_attention_mask
    data.attention_mask = align_mask  # 关键：对齐 Mask
    data.labels = labels

    return data


def main():
    logger.info("==========================================")
    logger.info("   Struct-MIF Model Integrity Check")
    logger.info("==========================================")

    # --- 配置 ---
    # 指向您本地的大模型路径 (相对路径或绝对路径均可)
    LOCAL_MODEL_PATH = "./pretrained_models/esm2_t33_650M_UR50D"

    # 1. 检查导入
    check_imports()

    # 2. 设备检查
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 3. 检查模型路径是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.error(f"❌ Local model path not found: {LOCAL_MODEL_PATH}")
        logger.error("Please verify the path or upload the model files.")
        sys.exit(1)
    else:
        logger.info(f"✅ Found local model at: {LOCAL_MODEL_PATH}")

    # 4. 初始化模型
    logger.info(">>> Step 2: Initializing Model (ESM-2 650M)...")
    try:
        from src.modeling.struct_mif import StructMIF
        from src.loss import MaskedMLMLoss

        # [关键修改]
        # ESM-2 650M hidden_size = 1280
        # 3Di embedding size = 128 (default)
        # gvp_node_in_dim = 1280 + 128 = 1408
        model = StructMIF(
            esm_model_name=LOCAL_MODEL_PATH,
            gvp_layers=2,  # 测试用，层数少一点初始化快
            gvp_node_in_dim=1408,  # <--- 必须是 1408，否则维度报错
            gvp_node_out_dim=64
        )
        model.to(device)
        logger.info("✅ Model initialized successfully.")

        # 打印参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"   Total Params: {total_params:,}")
        logger.info(f"   Trainable Params: {trainable_params:,} (Should be small, ESM is frozen)")

    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. 构造数据
    logger.info(">>> Step 3: Generating Dummy Data...")
    try:
        batch = get_dummy_batch(batch_size=2, seq_len=16, device=device)
        logger.info(f"✅ Dummy batch created.")

        # 验证对齐逻辑
        valid_res_count = batch.attention_mask.sum().item()
        graph_node_count = batch.x_3di.shape[0]
        if valid_res_count != graph_node_count:
            logger.error(
                f"❌ Alignment Mismatch! Valid Seq Tokens ({valid_res_count}) != Graph Nodes ({graph_node_count})")
            sys.exit(1)
        else:
            logger.info("   Alignment Check: PASS")

    except Exception as e:
        logger.error(f"❌ Data generation failed: {e}")
        sys.exit(1)

    # 6. 前向传播
    logger.info(">>> Step 4: Running Forward Pass...")
    try:
        # 启用混合精度
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(batch)

        logger.info(f"✅ Forward pass successful.")
        logger.info(f"   Output Logits: {logits.shape} (Should be [Total_Nodes, 33])")

    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. Loss 计算与反向传播
    logger.info(">>> Step 5: Running Backward Pass...")
    try:
        criterion = MaskedMLMLoss()

        # 计算 Loss
        # 只计算对齐部分的 loss
        masked_labels = batch.labels[batch.attention_mask.bool()]
        loss = criterion(logits, masked_labels)
        logger.info(f"   Loss value: {loss.item():.4f}")

        # 反向传播
        loss.backward()
        logger.info("✅ Backward pass successful.")

        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        if has_grad:
            logger.info("✅ Gradients computed successfully.")
        else:
            logger.warning("⚠️ No gradients found! Check if all parameters are frozen.")

    except Exception as e:
        logger.error(f"❌ Backward pass failed: {e}")
        sys.exit(1)

    logger.info("==========================================")
    logger.info("🎉 CONGRATULATIONS! Your model is healthy.")
    logger.info("==========================================")


if __name__ == "__main__":
    main()