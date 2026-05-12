import numpy as np
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBParser

# 忽略 Biopython 烦人的警告
warnings.simplefilter('ignore', BiopythonWarning)


class PDBFeatureExtractor:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        # 硬编码氨基酸映射，不依赖 Bio.PDB
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

        # --- 贪婪模式：遍历所有 Model, 所有 Chain ---
        # 对于 dompdb，通常只有一个 Model，但可能被切分成多个 Chain 片段
        # 我们把它们全部连起来
        for model in structure:
            for chain in model:
                for residue in chain:
                    # 1. 过滤非标准残基 (HETATM)
                    if residue.id[0] != ' ':
                        continue

                    # 2. 获取残基名并清洗
                    resname = residue.get_resname().strip()
                    if resname not in self.aa_map:
                        continue

                    # 3. 检查骨架原子完整性
                    if not all(atom in residue for atom in ['N', 'CA', 'C']):
                        continue

                    try:
                        # 4. 提取坐标
                        n = residue['N'].get_coord()
                        ca = residue['CA'].get_coord()
                        c = residue['C'].get_coord()

                        # 堆叠
                        res_coords = np.stack([n, ca, c])

                        coords.append(res_coords)
                        seq.append(self.aa_map[resname])

                    except Exception:
                        continue

            # 通常 PDB 只有一个 Model。读完第一个 Model 后退出，避免读取 NMR 的多个 Model
            break

            # 如果文件是空的或者没读到东西
        if len(coords) == 0:
            return None

        return {
            "coords": np.array(coords),  # [L, 3, 3]
            "seq": "".join(seq)  # String
        }


# 自测代码
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ext = PDBFeatureExtractor()
        res = ext.parse(sys.argv[1])
        if res:
            print(f"Parsed: {len(res['seq'])} residues.")
            print(f"Seq: {res['seq'][:50]}...")
        else:
            print("Failed.")