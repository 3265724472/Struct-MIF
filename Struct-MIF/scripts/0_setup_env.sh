#!/bin/bash
set -e  # 遇到错误立即停止

# ==============================================================================
# 配置区域
# ==============================================================================
PROJECT_ROOT=$(pwd)
DATA_BIN="$PROJECT_ROOT/data/bin"
MODEL_DIR="$PROJECT_ROOT/pretrained_models"
FOLDSEEK_URL="https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"

echo "====================================================================="
echo "   🚀 Struct-MIF 环境自动配置脚本"
echo "   工作目录: $PROJECT_ROOT"
echo "====================================================================="

# 1. 创建目录结构
echo "[1/4] Creating directory structure..."
mkdir -p "$DATA_BIN"
mkdir -p "$PROJECT_ROOT/data/raw_pdb"
mkdir -p "$PROJECT_ROOT/data/processed_graphs/train"
mkdir -p "$PROJECT_ROOT/data/processed_graphs/val"
mkdir -p "$PROJECT_ROOT/data/benchmarks"
mkdir -p "$PROJECT_ROOT/data/tmp"  # 临时文件存放处
mkdir -p "$MODEL_DIR"
mkdir -p "$PROJECT_ROOT/experiments"
mkdir -p "$PROJECT_ROOT/logs"

# 2. 安装 Foldseek (如果不存在)
echo "[2/4] Checking Foldseek..."
if [ -f "$DATA_BIN/foldseek" ]; then
    echo "    ✅ Foldseek already installed."
else
    echo "    ⬇️ Downloading Foldseek (Linux AVX2)..."
    wget -q --show-progress "$FOLDSEEK_URL" -O foldseek.tar.gz

    echo "    📦 Extracting..."
    tar -xzf foldseek.tar.gz

    echo "    🚚 Moving binary to data/bin/..."
    mv foldseek/bin/foldseek "$DATA_BIN/"
    chmod +x "$DATA_BIN/foldseek"

    # 清理
    rm foldseek.tar.gz
    rm -rf foldseek
    echo "    ✅ Foldseek installed successfully!"
fi

# 3. 安装 Python 依赖
# 注意: 这里假设你已经激活了 conda 环境 (比如 source activate protein_env)
echo "[3/4] Installing Python dependencies..."

# 3.1 安装 PyTorch (适配 RTX 4090, 推荐 CUDA 12.1)
# 如果已安装，pip 会跳过
echo "    --> Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3.2 安装 PyTorch Geometric (PyG) 及其依赖
# PyG 的安装比较挑版本，这里使用官方推荐的 wheel 方式
echo "    --> Installing PyTorch Geometric..."
pip install torch_geometric
# 安装可选依赖 (Scatter, Sparse) - 根据 PyTorch 版本动态调整
# 这里写死为 torch-2.1.0+cu121，如果您的 PyTorch 版本不同，请手动修改
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 3.3 安装项目特定依赖
echo "    --> Installing project libraries..."
pip install gvp-pytorch  # GVP 核心库
pip install transformers # ESM 模型
pip install biopython    # PDB 解析
pip install scipy pandas matplotlib tensorboard
pip install huggingface_hub # 用于下载模型

# 4. 下载预训练模型 (ESM-2-650M)
echo "[4/4] Downloading ESM-2 650M Model..."
if [ -d "$MODEL_DIR/esm2_t33_650M_UR50D" ]; then
    echo "    ✅ ESM-2 model seems to exist."
else
    echo "    ⬇️ Downloading from HuggingFace (This may take a while)..."
    # 使用 huggingface-cli 下载到指定目录
    huggingface-cli download facebook/esm2_t33_650M_UR50D \
        --local-dir "$MODEL_DIR/esm2_t33_650M_UR50D" \
        --local-dir-use-symlinks False
    echo "    ✅ Model downloaded."
fi

echo "====================================================================="
echo "🎉 Setup Complete! Ready to rock on RTX 4090."
echo "   Testing Foldseek..."
"$DATA_BIN/foldseek" --help | head -n 1
echo "====================================================================="