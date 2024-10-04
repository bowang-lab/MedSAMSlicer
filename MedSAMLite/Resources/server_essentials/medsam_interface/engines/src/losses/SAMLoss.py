from monai.losses import FocalLoss, DiceLoss
from torch import nn

from .components.IoULoss import IoULoss


class SAMLoss(nn.Module):
    """
    Loss function used in Segment Anything paper.
    """

    def __init__(
        self,
        dice_loss_weight=1.0,
        focal_loss_weight=20.0,
        iou_loss_weight=1.0,
    ):
        super().__init__()
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.focal_loss = FocalLoss(use_softmax=False, reduction="mean")
        self.iou_loss = IoULoss()

    def forward(self, pred_logits, pred_iou, gt_mask):
        """
        pred_logits: [B, 1, H, W]
        gt_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        dice_loss = self.dice_loss(pred_logits, gt_mask)
        focal_loss = self.focal_loss(pred_logits, gt_mask)
        iou_loss = self.iou_loss(pred_logits, pred_iou, gt_mask)
        return (
            self.dice_loss_weight * dice_loss
            + self.focal_loss_weight * focal_loss
            + self.iou_loss_weight * iou_loss
        )
