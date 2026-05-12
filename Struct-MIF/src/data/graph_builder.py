import torch
import numpy as np
import os
import logging
import warnings
import subprocess
import tempfile
import re
import shutil
import uuid
import gzip
from Bio.PDB import PDBParser
from Bio import BiopythonWarning
from scipy.spatial import cKDTree
from torch_geometric.data import Data

# 忽略 Biopython 警告
warnings.simplefilter('ignore', BiopythonWarning)


# =========================================================
# InternalPDBFeatureExtractor (保持稳健，支持 .gz)
# =========================================================
class InternalPDBFeatureExtractor:
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
        temp_pdb = None
        parse_path = pdb_path

        try:
            # 自动处理 .gz
            if pdb_path.endswith('.gz'):
                fd, temp_pdb = tempfile.mkstemp(suffix=".pdb")
                os.close(fd)
                with gzip.open(pdb_path, 'rb') as f_in:
                    with open(temp_pdb, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                parse_path = temp_pdb

            structure = self.parser.get_structure('structure', parse_path)

            coords = []
            seq = []
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
                break

            if len(coords) == 0: return None
            return {"coords": np.array(coords), "seq": "".join(seq)}

        except Exception:
            return None
        finally:
            if temp_pdb and os.path.exists(temp_pdb):
                os.remove(temp_pdb)


# =========================================================
# GraphBuilder 主类
# =========================================================
class GraphBuilder:
    def __init__(self, foldseek_bin_path, k_neighbors=30):
        self.foldseek_bin = foldseek_bin_path
        self.k = k_neighbors
        self.extractor = InternalPDBFeatureExtractor()

    def process(self, pdb_path):
        # 1. 解析 PDB
        pdb_data = self.extractor.parse(pdb_path)
        if pdb_data is None: return None

        coords = pdb_data['coords']  # shape: [L, 3, 3] (N, CA, C)
        seq = pdb_data['seq']
        length = len(seq)
        if length < 5: return None

        # 2. 运行 Foldseek (获取 3Di)
        seq_3di = self.run_foldseek(pdb_path)

        if seq_3di is None:
            # 如果 Foldseek 失败，这是一个严重错误，返回 None
            return None

        # 3. 长度对齐
        # Foldseek 生成的 3Di 长度可能与 PDB 解析的 AA 长度略有出入
        min_len = min(len(seq), len(seq_3di))

        # 截断对齐
        seq = seq[:min_len]
        seq_3di = seq_3di[:min_len]
        coords = coords[:min_len]

        # 再次检查长度 (防止截断后过短)
        if min_len < 5: return None

        # 4. 建图
        try:
            # 提取 CA 坐标用于建图
            ca_coords = coords[:, 1, :]

            # k-NN 建图
            tree = cKDTree(ca_coords)
            dists, idxs = tree.query(ca_coords, k=min(self.k, len(ca_coords)))

            src_list = []
            dst_list = []
            for i, neighbors in enumerate(idxs):
                for neighbor_idx in neighbors:
                    if i == neighbor_idx: continue
                    src_list.append(i)
                    dst_list.append(neighbor_idx)

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

            # [关键新增] 保存原始坐标 (pos)，用于后续 Dataset 计算 edge_vectors
            # pos 通常取 CA 原子的坐标
            pos = torch.tensor(ca_coords, dtype=torch.float)

            # Node Vectors (节点内部特征: N->CA, CA->C)
            node_coords_th = torch.tensor(coords, dtype=torch.float)
            n_vec = node_coords_th[:, 1] - node_coords_th[:, 0]
            c_vec = node_coords_th[:, 2] - node_coords_th[:, 1]
            node_vectors = torch.stack([n_vec, c_vec], dim=1)

            # 封装 Data 对象
            data = Data(
                num_nodes=min_len,
                edge_index=edge_index,
                node_vectors=node_vectors,
                pos=pos,  # <--- 必须包含此字段！
                seq=seq,
                seq_3di=seq_3di
            )
            return data
        except Exception:
            return None

    def run_foldseek(self, pdb_path):
        """
        [Plan C] 使用 easy-search 自比对来提取 3Di 序列
        命令：foldseek easy-search input.pdb input.pdb result.m8 tmp --format-output q3di
        """
        unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_pdb = os.path.join(temp_dir, f"in_{unique_id}.pdb")
            result_m8 = os.path.join(temp_dir, "result.m8")
            tmp_internal = os.path.join(temp_dir, "tmp")

            try:
                # 1. 准备输入文件 (支持 .gz)
                if pdb_path.endswith('.gz'):
                    with gzip.open(pdb_path, 'rb') as f_in:
                        with open(temp_input_pdb, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy(pdb_path, temp_input_pdb)

                # 2. 运行 easy-search
                # 这是一个 hack：让蛋白质和它自己比对。
                # --format-output q3di : 明确要求只输出 Query 的 3Di 序列
                # --exhaustive-search 1 : 确保一定能搜到自己
                cmd = [
                    self.foldseek_bin, "easy-search",
                    temp_input_pdb, temp_input_pdb,  # query = target
                    result_m8,
                    tmp_internal,
                    "--format-output", "q3di",
                    "--exhaustive-search", "1",
                    "-v", "0"  # 静默模式
                ]

                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 3. 读取结果
                if os.path.exists(result_m8):
                    with open(result_m8, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # 结果可能有多行（如果有多个比对），我们只取第一行（它是最佳匹配，即自己）
                            # 去掉换行符，且去掉可能存在的空字符
                            seq_3di = lines[0].strip()
                            if seq_3di:
                                return seq_3di

            except Exception:
                return None
        return None