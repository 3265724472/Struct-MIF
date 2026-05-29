import torch
import torch.nn as nn


class MaskedMLMLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        # 关键：设置 ignore_index=-100
        # 这样 dataset 里那些没被 mask 的位置（值为-100）就不会参与 Loss 计算
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        # logits: [Batch * Nodes, Vocab]
        # labels: [Batch * Nodes]

        # 确保维度匹配
        if labels.dim() > 1:
            labels = labels.view(-1)

        # 自动计算 CrossEntropy，忽略 label=-100 的位置
        return self.criterion(logits, labels)