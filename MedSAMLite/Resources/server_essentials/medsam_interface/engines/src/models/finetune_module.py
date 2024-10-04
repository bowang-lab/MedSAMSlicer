from typing import Any, Dict

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.losses import SAMLoss
from src.metrics.generalized_dice import GeneralizedDiceMetric
from src.models.base_sam import BaseSAM


class FinetuneLitModule(LightningModule):

    def __init__(
        self,
        model: BaseSAM,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module = None,
        scheduler_interval: str = "step",
        freeze_image_encoder: bool = False,
        freeze_prompt_encoder: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model", "criterion"])

        self.model = model
        self.criterion = criterion if criterion is not None else SAMLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc = GeneralizedDiceMetric()
        self.test_acc = GeneralizedDiceMetric()

        if freeze_image_encoder:
            self.model.image_encoder.requires_grad_(False)
            self.model.image_encoder.eval()
        if freeze_prompt_encoder:
            self.model.prompt_encoder.requires_grad_(False)
            self.model.prompt_encoder.eval()

    def model_step(self, batch, metric=None):
        imgs = batch["image"]  # (B, 3, H, W)
        image_encoder_input_size = max(imgs.shape[-2:])
        target_masks = (
            batch["masks"]
            .view(-1, 1, image_encoder_input_size, image_encoder_input_size)
            .float()
        )  # (B * N, 1, H, W)
        boxes = batch["boxes"].view(-1, 4)  # (B * N, 4)

        image_embeddings = self.model.image_encoder(imgs)  # (B, 256, 64, 64)
        masks, iou_preds = self.model.prompt_and_decoder(image_embeddings, boxes)
        masks = self.model.postprocess_masks(
            masks=masks,
            input_size=(image_encoder_input_size, image_encoder_input_size),
            original_size=(-1, -1),
            return_with_image_encoder_size=True,
        )  # (B * N, 1, H, W)
        loss = self.criterion(
            pred_logits=masks,
            pred_iou=iou_preds,
            gt_mask=target_masks,
        )

        if metric is not None:
            metric.update(preds=(masks > 0), gts=target_masks)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self.model_step(batch, metric=self.val_acc)
        self.val_loss(loss)
        metrics = {"val/loss": self.val_loss, "val/acc": self.val_acc}
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        loss = self.model_step(batch, metric=self.test_acc)
        self.test_loss(loss)
        metrics = {"test/loss": self.test_loss, "test/acc": self.test_acc}
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FinetuneLitModule(None, None, None)
