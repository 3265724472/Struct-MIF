import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


class FoldseekRunner:
    """
    Foldseek 命令行调用封装
    用于生成 3Di 序列 (structure sequence)
    """

    def __init__(self, binary_path: str = "./data/bin/foldseek"):
        self.binary_path = binary_path
        if not os.path.exists(binary_path):
            # 尝试在系统路径查找
            import shutil
            if shutil.which("foldseek"):
                self.binary_path = "foldseek"
            else:
                logger.warning(f"Foldseek binary not found at {binary_path}. Please setup_env.sh")

    def run(self, pdb_path: str) -> str:
        """
        输入 PDB 路径，返回 3Di 字符串
        """
        # 创建临时输出文件
        # Foldseek 需要输出文件路径作为参数
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # 命令: foldseek structureto3didescriptor <input.pdb> <output.tsv>
            cmd = [
                self.binary_path,
                "structureto3didescriptor",
                pdb_path,
                output_path
            ]

            # 运行命令，静默输出
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # 解析结果
            return self._parse_output(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"Foldseek execution failed for {pdb_path}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error running Foldseek: {e}")
            return ""
        finally:
            # 清理临时文件
            if os.path.exists(output_path):
                os.remove(output_path)

    def _parse_output(self, tsv_path: str) -> str:
        """解析 structureto3didescriptor 的输出 TSV"""
        with open(tsv_path, 'r') as f:
            lines = f.readlines()
            # 典型的 Foldseek 输出包含 header 和 data
            # 格式通常是: query, target, 3di_sequence
            # 我们只需要最后一行，最后一列
            if len(lines) == 0:
                return ""

            # 取最后一行数据
            last_line = lines[-1].strip()
            parts = last_line.split('\t')

            # 通常 3Di 序列是第3列 (index 2)
            # 有些版本可能是 headerless，需要根据实际情况防御性编程
            if len(parts) >= 3:
                return parts[2]
            else:
                logger.warning(f"Unexpected Foldseek output format: {last_line}")
                return ""