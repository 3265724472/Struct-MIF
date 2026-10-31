import os
import sys
import argparse
import glob
import multiprocessing
import logging
from tqdm import tqdm
from Bio import Align
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# 添加项目根目录到 Path
sys.path.append(os.getcwd())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_leakage.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LeakageDetector:
    def __init__(self, threshold=0.3):
        """
        Args:
            threshold: 序列一致性阈值 (0.0 - 1.0).
                       通常 < 0.3 (30%) 被认为是安全的 Zero-shot 划分 (Twilight Zone).
        """
        self.threshold = threshold
        self.aligner = Align.PairwiseAligner()
        # 配置对齐参数 (全局对齐)
        self.aligner.mode = 'global'
        self.aligner.match_score = 1.0
        self.aligner.mismatch_score = 0.0
        self.aligner.open_gap_score = -0.5
        self.aligner.extend_gap_score = -0.1

    def calculate_identity(self, seq1, seq2):
        """计算两个序列的一致性"""
        if not seq1 or not seq2:
            return 0.0

        # 简单优化: 如果长度差异太大，一致性不可能高
        len1, len2 = len(seq1), len(seq2)
        max_len = max(len1, len2)
        min_len = min(len1, len2)

        if min_len / max_len < self.threshold:
            return 0.0

        # 执行对齐
        score = self.aligner.score(seq1, seq2)

        # Identity = Matches / Max_Length
        # 这里的 score 近似于 Matches (因为 mismatch=0)
        identity = score / max_len
        return identity


def get_pdb_sequence(pdb_path):
    """
    从 PDB 文件提取氨基酸序列
    为了速度，只读取第一条链的 ATOM 记录
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('temp', pdb_path)
        for model in structure:
            for chain in model:
                seq = []
                for residue in chain:
                    if residue.id[0] != ' ': continue  # 跳过 HETATM
                    try:
                        seq.append(seq1(residue.get_resname()))
                    except:
                        seq.append('X')
                return "".join(seq)  # 只返回第一条链
    except Exception as e:
        return None


def check_worker(args):
    """
    多进程 Worker: 检查一个训练样本是否与任何测试样本冲突
    """
    train_path, test_sequences, threshold = args

    # 1. 获取训练样本序列
    train_seq = get_pdb_sequence(train_path)
    if not train_seq:
        return None  # 文件损坏

    detector = LeakageDetector(threshold)
    leaks = []

    # 2. 与所有测试样本比对
    for test_name, test_seq in test_sequences.items():
        identity = detector.calculate_identity(train_seq, test_seq)
        if identity > threshold:
            leaks.append({
                'train_file': os.path.basename(train_path),
                'test_benchmark': test_name,
                'identity': identity
            })
            # 只要发现与任一测试集冲突，就可以判定该训练样本不合格，提前退出
            break

    return leaks


def main():
    parser = argparse.ArgumentParser(description="检测训练集与测试集之间的数据泄漏 (Data Leakage)")

    parser.add_argument("--train_dir", type=str, default="./data/raw_pdb",
                        help="训练集 PDB 目录")
    parser.add_argument("--test_dir", type=str, default="./data/benchmarks",
                        help="测试集目录 (包含 .pdb 文件)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="相似度阈值 (默认 0.3, 即 30%)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="CPU 进程数")
    parser.add_argument("--delete_leaks", action="store_true",
                        help="[危险] 自动删除检测到的泄漏文件")

    args = parser.parse_args()

    # 1. 加载所有测试集序列 (Benchmarks)
    # 测试集通常很小，直接全部读入内存
    logger.info(f"Loading benchmarks from {args.test_dir}...")
    test_pdbs = glob.glob(os.path.join(args.test_dir, "**/*.pdb"), recursive=True)
    test_sequences = {}  # {filename: sequence}

    for p in test_pdbs:
        seq = get_pdb_sequence(p)
        if seq:
            name = os.path.basename(p)
            test_sequences[name] = seq
            logger.info(f"  Benchmark loaded: {name} (Len={len(seq)})")

    if not test_sequences:
        logger.error("No benchmark PDBs found! Please check --test_dir.")
        return

    # 2. 扫描训练集文件
    logger.info(f"Scanning training PDBs in {args.train_dir}...")
    train_pdbs = glob.glob(os.path.join(args.train_dir, "*.pdb"))
    if not train_pdbs:
        logger.error("No training PDBs found!")
        return

    logger.info(f"Checking {len(train_pdbs)} training files against {len(test_sequences)} benchmarks...")
    logger.info(f"Identity Threshold: {args.threshold * 100}%")

    # 3. 多进程比对
    # 构造任务列表
    tasks = [(p, test_sequences, args.threshold) for p in train_pdbs]

    leaked_files = []

    with multiprocessing.Pool(args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(check_worker, tasks), total=len(tasks)):
            if result:
                leaked_files.extend(result)

    # 4. 报告结果
    logger.info("============================================")
    if len(leaked_files) == 0:
        logger.info("✅ No data leakage detected. Your split is clean!")
    else:
        logger.warning(f"🚨 FOUND {len(leaked_files)} LEAKING PROTEINS!")
        logger.warning(f"These training samples are >{args.threshold * 100}% identical to test set.")

        # 保存黑名单
        blacklist_path = "leaked_blacklist.txt"
        with open(blacklist_path, "w") as f:
            f.write("Train_File\tTest_Target\tIdentity\n")
            for leak in leaked_files:
                f.write(f"{leak['train_file']}\t{leak['test_benchmark']}\t{leak['identity']:.4f}\n")
                logger.warning(f"  Leak: {leak['train_file']} <-> {leak['test_benchmark']} ({leak['identity']:.2%})")

        logger.info(f"Blacklist saved to {blacklist_path}")

        # 自动删除 (如果开启)
        if args.delete_leaks:
            logger.info("Deleting leaked files...")
            count = 0
            for leak in leaked_files:
                full_path = os.path.join(args.train_dir, leak['train_file'])
                if os.path.exists(full_path):
                    os.remove(full_path)

                    # 同时尝试删除对应的预处理 .pt 文件
                    pt_name = leak['train_file'].replace(".pdb", ".pt")
                    pt_path = os.path.join("./data/processed_graphs/train", pt_name)
                    if os.path.exists(pt_path):
                        os.remove(pt_path)
                    count += 1
            logger.info(f"Deleted {count} files.")
        else:
            logger.info("Run with --delete_leaks to remove these files automatically.")

    logger.info("============================================")


if __name__ == "__main__":
    main()