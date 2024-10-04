from pathlib import Path

import torch
import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.models.base_sam import BaseSAM
from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def export(cfg: DictConfig):
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseSAM = hydra.utils.instantiate(cfg.model).to("cpu")
    model.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.prompt_encoder.input_image_size = (512, 512)
    torch.save(model, output_dir / "model.pth")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="export_torch.yaml"
)
def main(cfg: DictConfig):
    """Main entry point for exporting.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    export(cfg)


if __name__ == "__main__":
    main()
