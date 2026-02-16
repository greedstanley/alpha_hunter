import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import warnings
import numpy as np

class FocalLoss(nn.Module):
    """
    SOTA 的損失函數，用來對付佔了 70% 的 Hold (盤整) 標籤。
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # 如果給定，應該是長度為 num_classes 的 Tensor
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, num_classes] (Logits)
        # targets: [Batch] (整數標籤 0, 1, 2)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_mcc(preds_logits, targets):
    """
    計算 Matthews Correlation Coefficient (你的北極星指標)
    
    Args:
        preds_logits: 模型輸出的 Logits
        targets: 真實標籤
    """
    preds = torch.argmax(preds_logits, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # [Fix] Handle cases where only one class is present to suppress sklearn warnings
    # If there's only 1 class in truth or pred, MCC is not well-defined (denominator is 0)
    # sklearn returns 0.0 in this case but warns. We suppress the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcc = matthews_corrcoef(targets, preds)
        
    return mcc


