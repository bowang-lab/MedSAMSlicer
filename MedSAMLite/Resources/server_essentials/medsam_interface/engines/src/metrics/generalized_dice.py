import torch
from torchmetrics import Metric
from monai.metrics import compute_generalized_dice


class GeneralizedDiceMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("dsc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, gts: torch.Tensor):
        dsc = compute_generalized_dice(preds, gts)
        self.dsc += dsc.sum()
        self.total += dsc.numel()

    def compute(self) -> torch.Tensor:
        return self.dsc.float() / self.total
