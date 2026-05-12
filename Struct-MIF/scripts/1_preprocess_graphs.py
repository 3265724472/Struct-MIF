import sys
import os
import argparse
import logging
import multiprocessing
import glob
import torch
import numpy as np
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from tqdm import tqdm
from functools import partial

# --- [核弹补丁区域] 开始 ---
# 我们不信任环境里的 pdb_parser.py 了，直接把修正后的类定义在这里
# 这样所有子进程都必须使用这份代码

warnings.simplefilter('ignore', BiopythonWarning)


class SafePDBFeatureExtractor:
    """
    [本地修正版] 贪婪模式 PDB 解析器
    直接定义在脚本里，防止 import 到旧代码
    """

    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            'MSE': 'M', 'SEC': 'C'
        }

    def parse(self, pdb_path):
        return self.parse_pdb(pdb_path)

    def parse_pdb(self, pdb_path):
        try:
            structure = self.parser.get_structure('structure', pdb_path)
        except Exception:
            return None

        coords = []
        seq = []

        # 贪婪模式：读完所有 Model 所有 Chain
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ': continue
                    resname = residue.get_resname().strip()
                    if resname not in self.aa_map: continue
                    if not all(atom in residue for atom in ['N', 'CA', 'C']): continue

                    try:
                        n = residue['N'].get_coord()
                        ca = residue['CA'].get_coord()
                        c = residue['C'].get_coord()
                        res_coords = np.stack([n, ca, c])
                        coords.append(res_coords)
                        seq.append(self.aa_map[resname])
                    except Exception:
                        continue
            # 只取第一个 Model，但要读完 Model 内所有 Chain
            break

        if len(coords) == 0: return None
        return {"coords": np.array(coords), "seq": "".join(seq)}


# --- 强制执行猴子补丁 (Monkey Patch) ---
# 这几行代码会把 src 库里可能存在的“旧类”直接替换成上面的“新类”
# 确保 import src.data.graph_builder 时，它用的是我们的 SafePDBFeatureExtractor

# 1. 先导入目标模块
try:
    import src.common.pdb_parser
    import src.data.graph_builder

    # 2. 暴力覆盖
    print("💉 INJECTING: Monkey patching PDBFeatureExtractor with local code...")
    src.common.pdb_parser.PDBFeatureExtractor = SafePDBFeatureExtractor
    src.data.graph_builder.PDBFeatureExtractor = SafePDBFeatureExtractor
    print("✅ INJECTING: Patch successful.")
except ImportError as e:
    print(f"⚠️ Warning during patching: {e}")
    # 如果导入失败，我们会在 process_single_pdb 里再次尝试
    pass

from src.data.graph_builder import GraphBuilder

# --- [核弹补丁区域] 结束 ---


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PDB Preprocess")
    parser.add_argument("--input_dir", type=str, default="./data/raw_pdb")
    parser.add_argument("--output_dir", type=str, default="./data/processed_graphs/train")
    parser.add_argument("--foldseek_bin", type=str, default="./data/bin/foldseek")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--k_neighbors", type=int, default=30)
    return parser.parse_args()


def process_single_pdb(file_info, foldseek_bin, k_neighbors):
    """单个处理函数"""
    pdb_path, save_path = file_info
    if os.path.exists(save_path): return "skipped"

    try:
        # 双重保险：在子进程里再次确认使用了补丁类
        # 即使 import 机制在子进程重置了，我们在这里实例化 GraphBuilder 之前
        # 再次强行修改它的依赖
        import src.data.graph_builder
        src.data.graph_builder.PDBFeatureExtractor = SafePDBFeatureExtractor

        builder = GraphBuilder(foldseek_bin_path=foldseek_bin, k_neighbors=k_neighbors)
        graph_data = builder.process(pdb_path)

        if graph_data is None: return f"failed: {pdb_path}"
        torch.save(graph_data, save_path)
        return "success"
    except Exception as e:
        return f"error: {pdb_path} - {str(e)}"


def main():
    args = parse_args()
    if not os.path.exists(args.input_dir): return
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists("logs"): os.makedirs("logs")

    logger.info(f"Scanning PDB files in {args.input_dir}...")
    pdb_files = glob.glob(os.path.join(args.input_dir, "*.pdb"))
    if not pdb_files: pdb_files = glob.glob(os.path.join(args.input_dir, "*.cif"))

    total_files = len(pdb_files)
    logger.info(f"Found {total_files} structures. Starting with {args.num_workers} workers...")

    tasks = []
    for p in pdb_files:
        s = os.path.join(args.output_dir, os.path.splitext(os.path.basename(p))[0] + ".pt")
        tasks.append((p, s))

    worker_fn = partial(process_single_pdb, foldseek_bin=args.foldseek_bin, k_neighbors=args.k_neighbors)

    success, skipped, failed = 0, 0, 0
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(worker_fn, tasks), total=total_files):
            if res == "success":
                success += 1
            elif res == "skipped":
                skipped += 1
            else:
                failed += 1
                logger.warning(res)

    logger.info("Processing Complete!")
    logger.info(f"Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()