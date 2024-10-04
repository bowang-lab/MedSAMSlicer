import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_logits, pred_iou, gt_mask):
        """
        pred_mask: [B, 1, H, W]
        gt_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_logits.shape == gt_mask.shape
        assert pred_logits.shape[1] == 1

        pred_mask = pred_logits > 0
        reduce_axis = list(range(2, len(pred_logits.shape)))
        intersection = torch.sum(pred_mask * gt_mask, dim=reduce_axis)
        union = (
            torch.sum(pred_mask, dim=reduce_axis)
            + torch.sum(gt_mask, dim=reduce_axis)
            - intersection
        )
        iou = intersection / (union + self.eps)
        return torch.mean((iou - pred_iou) ** 2)
