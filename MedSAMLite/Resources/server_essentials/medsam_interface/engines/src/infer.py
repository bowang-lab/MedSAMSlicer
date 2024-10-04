from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import hydra
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm
from matplotlib import pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch.utils.data import DataLoader

from src.models.base_sam import BaseSAM
from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)
from src.utils.transforms import get_bbox, resize_box
from src.utils.visualize import visualize_output


log = RankedLogger(__name__, rank_zero_only=True)


def infer_2D(model: BaseSAM, data, device, output_dir: Path, save_overlay: bool):
    npz_name = data["npz_name"]
    img = data["image"].unsqueeze(0).to(device)
    boxes = data["boxes"].to(device)
    original_size = data["original_size"].tolist()
    new_size = data["new_size"].tolist()

    image_embedding = model.image_encoder(img)
    masks, _ = model.prompt_and_decoder(image_embedding, boxes)
    masks = model.postprocess_masks(masks, new_size, original_size)
    masks = masks.squeeze(1).cpu().numpy()

    segs = np.zeros(original_size, dtype=np.uint16)
    for idx in range(len(boxes)):
        segs[masks[idx] > 0] = idx + 1

    np.savez_compressed(output_dir / "npz" / npz_name, segs=segs)

    # visualize image, mask and bounding box
    if save_overlay:
        visualize_output(
            img=data["original_image"],
            boxes=data["original_boxes"],
            segs=segs,
            save_file=(output_dir / "png" / npz_name).with_suffix(".png"),
        )


def infer_3D(model: BaseSAM, data, device, output_dir: Path, save_overlay: bool):
    npz_name = data["npz_name"]
    imgs = data["image"].to(device)  # (D, 3, H, W)
    boxes = data["boxes"]  # (N, 6), [[x_min, y_min, z_min, x_max, y_max, z_max]]
    original_size = data["original_size"].tolist()  # (2)
    new_size = data["new_size"].tolist()  # (2)
    prompt_encoder_input_size = data["prompt_encoder_input_size"]

    segs = np.zeros((imgs.shape[0], *original_size), dtype=np.uint16)

    for idx, box3D in enumerate(boxes, start=1):
        segs_i = np.zeros_like(segs, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        z_min = max(z_min, 0)
        z_max = min(z_max, imgs.shape[0])
        box_default = np.array([x_min, y_min, x_max, y_max])
        z_middle = (z_max + z_min) // 2

        # infer from middle slice to the z_max
        box_2D = box_default
        for z in range(int(z_middle), int(z_max)):
            img_2d = imgs[z, :, :, :].unsqueeze(0)  # (1, 3, H, W)
            image_embedding = model.image_encoder(img_2d)  # (1, 256, 64, 64)

            box_torch = torch.as_tensor(
                box_2D[None, ...], dtype=torch.float, device=device
            )  # (B, 4)
            mask, _ = model.prompt_and_decoder(image_embedding, box_torch)
            mask = model.postprocess_masks(mask, new_size, original_size)
            mask = mask.squeeze().cpu().numpy()
            if np.max(mask) > 0:
                box_2D = get_bbox(mask)
                box_2D = resize_box(
                    box=box_2D,
                    original_size=original_size,
                    prompt_encoder_input_size=prompt_encoder_input_size,
                )
                segs_i[z, mask > 0] = 1
            else:
                box_2D = box_default

        # infer from middle slice to the z_min
        if np.max(segs_i[int(z_middle), :, :]) == 0:
            box_2D = box_default
        else:
            box_2D = get_bbox(segs_i[int(z_middle), :, :])
            box_2D = resize_box(
                box=box_2D,
                original_size=original_size,
                prompt_encoder_input_size=prompt_encoder_input_size,
            )

        for z in range(int(z_middle - 1), int(z_min - 1), -1):
            img_2d = imgs[z, :, :, :].unsqueeze(0)  # (1, 3, H, W)
            image_embedding = model.image_encoder(img_2d)  # (1, 256, 64, 64)

            box_torch = torch.as_tensor(
                box_2D[None, ...], dtype=torch.float, device=device
            )  # (B, 4)
            mask, _ = model.prompt_and_decoder(image_embedding, box_torch)
            mask = model.postprocess_masks(mask, new_size, original_size)
            mask = mask.squeeze().cpu().numpy()
            if np.max(mask) > 0:
                box_2D = get_bbox(mask)
                box_2D = resize_box(
                    box=box_2D,
                    original_size=original_size,
                    prompt_encoder_input_size=prompt_encoder_input_size,
                )
                segs_i[z, mask > 0] = 1
            else:
                box_2D = box_default

        segs[segs_i > 0] = idx

    np.savez_compressed(output_dir / "npz" / npz_name, segs=segs)

    # visualize image, mask and bounding box
    if save_overlay:
        z = segs.shape[0] // 2
        visualize_output(
            img=data["original_image"][z],
            boxes=data["original_boxes"][:, [0, 1, 3, 4]],
            segs=segs[z],
            save_file=(output_dir / "png" / npz_name).with_suffix(".png"),
        )


@task_wrapper
@torch.no_grad()
def infer(cfg: DictConfig):
    log.info(f"Instantiating dataloader <{cfg.data._target_}>")
    dataloader: DataLoader = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseSAM = hydra.utils.instantiate(cfg.model).to(cfg.device)
    model = torch.compile(model)
    model.eval()

    output_dir = Path(cfg.output_dir)
    (output_dir / "npz").mkdir(parents=True, exist_ok=True)
    if cfg.save_overlay:
        (output_dir / "png").mkdir(parents=True, exist_ok=True)

    for data in tqdm(dataloader):
        start_time = time()
        if data["image_type"] == "2D":
            infer_2D(model, data, cfg.device, output_dir, cfg.save_overlay)
        elif data["image_type"] == "3D":
            infer_3D(model, data, cfg.device, output_dir, cfg.save_overlay)
        else:
            raise NotImplementedError("Only support 2D and 3D image")
        end_time = time()
        print(f"Predicted {data['npz_name']} in {end_time - start_time:.2f}s")


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    infer(cfg)


if __name__ == "__main__":
    main()
