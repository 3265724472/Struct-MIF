import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig


class ESMWrapper(nn.Module):
    """
    ESM-2 包装器 (冻结版)

    职责:
    1. 加载预训练权重
    2. 彻底冻结参数 (显存优化)
    3. 提供简洁的特征提取接口
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        print(f"Loading Semantic Encoder: {model_name}...")

        # 优化: 可以在这里加入 torch_dtype=torch.float16 以进一步减半显存
        # self.model = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model = EsmModel.from_pretrained(model_name)

        # 获取输出维度 (通常是 1280)
        self.hidden_size = self.model.config.hidden_size

        # 核心: 初始化即冻结
        self._freeze()

    def _freeze(self):
        """冻结所有参数，并不存储梯度信息"""
        self.model.eval()  # 切换到评估模式 (关闭 Dropout 等)
        for param in self.model.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """
        重写 train 方法。
        确保无论外部如何调用 model.train()，ESM 内部始终保持 eval 模式。
        这是防止 BatchNorm/Dropout 在微调时失效的关键。
        """
        super().train(mode)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [Batch, Seq_Len]
            attention_mask: [Batch, Seq_Len]

        Returns:
            last_hidden_state: [Batch, Seq_Len, Hidden_Dim]
        """
        # 显式使用 no_grad 确保中间变量不占用显存
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False
            )
        return outputs.last_hidden_state