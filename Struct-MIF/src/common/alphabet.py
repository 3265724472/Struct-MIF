class EsmAlphabet:
    """
    ESM-2 标准词表映射
    参考: facebook/esm2_t33_650M_UR50D 的 vocab.txt
    """

    def __init__(self):
        # 标准 tokens
        self.cls_token = "<cls>"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"

        # 特殊 token ID (ESM 标准)
        self.cls_idx = 0
        self.pad_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # 标准氨基酸 (顺序通常如下，但在使用 Tokenizer 时应以 tokenizer.json 为准)
        # 这里列出仅供参考或手动映射使用
        self.standard_toks = [
            "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
            "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
            "X", "B", "U", "Z", "O", ".", "-"
        ]

        # <mask> 的 ID 通常在列表末尾附近，具体取决于版本
        # 建议主要使用 EsmTokenizer.from_pretrained 加载
        # 此类主要用于存储特殊 token 的 ID 常量


class FoldseekAlphabet:
    """
    Foldseek 3Di 结构词表
    共有 20 种结构状态，我们通常映射到 1-20，0 留给 Padding
    """

    def __init__(self):
        # 20 种 3Di 状态字符
        self.tokens = [
            'd', 'p', 'v', 'l', 'i', 'a', 'k', 'e', 'g', 's',
            'h', 't', 'r', 'f', 'y', 'w', 'm', 'c', 'n', 'q'
        ]

        # 字符 -> ID 映射 (1-based index, 0 is padding)
        self.tok_to_idx = {tok: i + 1 for i, tok in enumerate(self.tokens)}
        self.idx_to_tok = {i + 1: tok for i, tok in enumerate(self.tokens)}

        self.pad_idx = 0
        self.vocab_size = len(self.tokens) + 1  # 21

    def encode(self, seq_str: str):
        """将 3Di 字符串转换为 ID 列表"""
        return [self.tok_to_idx.get(c.lower(), self.pad_idx) for c in seq_str]

    def decode(self, idx_list):
        """将 ID 列表转回字符串"""
        return "".join([self.idx_to_tok.get(idx, "?") for idx in idx_list])


# 单例实例，方便外部导入使用
esm_alphabet = EsmAlphabet()
foldseek_alphabet = FoldseekAlphabet()