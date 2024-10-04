from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


class DistillLitModule(LightningModule):

    def __init__(
        self,
        student_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        teacher_net: Optional[torch.nn.Module] = None,
        scheduler_interval: str = "epoch",
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["student_net", "teacher_net"])

        self.student_encoder = (
            student_net.image_encoder
            if "image_encoder" in dir(student_net)
            else student_net
        )
        if teacher_net is None:
            self.teacher_encoder = None
        else:
            self.teacher_encoder = (
                teacher_net.image_encoder
                if "image_encoder" in dir(teacher_net)
                else teacher_net
            )
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()

        self.criterion = torch.nn.MSELoss()
        self.train_loss = MeanMetric()

    def model_step(self, batch):
        student_embeddings = self.student_encoder(batch["image"])
        if self.teacher_encoder is not None:
            teacher_embeddings = self.teacher_encoder(batch["teacher_image"])
        else:
            teacher_embeddings = batch["embedding"]
        loss = self.criterion(student_embeddings, teacher_embeddings)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        pass

    def test_step(self, batch, batch_idx) -> None:
        pass

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
    _ = DistillLitModule(None, None, None)
