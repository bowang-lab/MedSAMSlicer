### Code and models are adopted from https://github.com/hieplpvip/medficientsam
### `engines.src` is almost an identical copy of https://github.com/hieplpvip/medficientsam/tree/59504938bb37ab7e2832ede358051976e740efe5/src

import argparse
import sys
from datetime import datetime
from glob import glob
from os.path import join, basename
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent))

class ResizeLongestSide(torch.nn.Module):
    def __init__(
        self,
        long_side_length: int,
        interpolation: str,
    ) -> None:
        super().__init__()
        self.long_side_length = long_side_length
        self.interpolation = interpolation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        oldh, oldw = image.shape[-2:]
        if max(oldh, oldw) == self.long_side_length:
            return image
        newh, neww = self.get_preprocess_shape(oldh, oldw, self.long_side_length)
        return F.interpolate(
            image, (newh, neww), mode=self.interpolation, align_corners=False
        )

    @staticmethod
    def get_preprocess_shape(
        oldh: int,
        oldw: int,
        long_side_length: int,
    ) -> Tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class MinMaxScale(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) >= 3 and image.shape[-3] == 3
        min_val = image.amin((-3, -2, -1), keepdim=True)
        max_val = image.amax((-3, -2, -1), keepdim=True)
        return (image - min_val) / torch.clip(max_val - min_val, min=1e-8, max=None)


class PadToSquare(torch.nn.Module):
    def __init__(self, target_size: int) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        return F.pad(image, (0, self.target_size - w, 0, self.target_size - h), value=0)


def get_bbox(mask: np.ndarray) -> np.ndarray:
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes


def resize_box(
    box: np.ndarray,
    original_size: Tuple[int, int],
    prompt_encoder_input_size: int,
) -> np.ndarray:
    new_box = np.zeros_like(box)
    ratio = prompt_encoder_input_size / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


def get_image_transform(
    long_side_length: int,
    min_max_scale: bool = True,
    normalize: bool = False,
    pixel_mean: Optional[List[float]] = None,
    pixel_std: Optional[List[float]] = None,
    interpolation: str = "bilinear",
) -> transforms.Transform:
    tsfm = [
        ResizeLongestSide(long_side_length, interpolation),
        transforms.ToDtype(dtype=torch.float32, scale=False),
    ]
    if min_max_scale:
        tsfm.append(MinMaxScale())
    if normalize:
        tsfm.append(transforms.Normalize(pixel_mean, pixel_std))
    tsfm.append(PadToSquare(long_side_length))
    return transforms.Compose(tsfm)


class MedficientSAMCore:
    model = None
    device = None
    MedSAM_CKPT_PATH = None

    H = None
    W = None
    image_shape = None
    embeddings = None

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = torch.load(self.MedSAM_CKPT_PATH, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_progress(self):
        return {'layers': 100 if self.image_shape is None else self.image_shape[0], 'generated_embeds': len(self.embeddings)}

    def set_image(self, image_data, wmin, wmax, zmin, zmax, recurrent_func):
        self.embeddings = []

        self.image_shape = image_data.shape
        self.original_size = image_data.shape[-2:]

        if len(image_data.shape) == 3:
            # gray: (D, H, W) -> (D, 3, H, W)
            tsfm_img_3D = np.repeat(image_data[:, None, ...], 3, axis=1)
        else:
            # rgb: (D, H, W, 3) -> (D, 3, H, W)
            tsfm_img_3D = np.transpose(image_data, (0, 3, 1, 2))

        transform_image = get_image_transform(long_side_length=512)
        tsfm_img_3D = transform_image(torch.tensor(tsfm_img_3D, dtype=torch.uint8))

        for z in range(image_data.shape[0]):
            if recurrent_func is not None:
                recurrent_func()
            image_embedding = None
            calculation_condition = (zmax == -1) or ((zmin-1) <= z <= (zmax+1)) # Full embedding or partial embedding that lies between slices
            if calculation_condition:
                img_2d = tsfm_img_3D[z, :, :, :].unsqueeze(0).to(self.device)  # (1, 3, H, W)
                image_embedding = self.model.image_encoder(img_2d).detach()  # (1, 256, 64, 64)
            else:
                image_embedding = None
            if image_embedding is not None:
              print(image_embedding.shape, image_embedding.dtype)
            self.embeddings.append(image_embedding)#.detach().cpu().numpy())

    @torch.no_grad()
    def infer(self, slice_idx, bbox, zrange):
        res = {}

        new_size = ResizeLongestSide.get_preprocess_shape(
            self.original_size[0], self.original_size[1], 512
        )
        prompt_encoder_input_size = self.model.prompt_encoder.input_image_size[0]

        z_min, z_max = zrange
        z_max = min(z_max+1, len(self.embeddings))
        z_min = max(z_min-1, 0)
        x_min, y_min, x_max, y_max = bbox

        box2D = np.array([x_min, y_min, x_max, y_max])
        box2D = resize_box(
            box2D,
            original_size=self.original_size,
            prompt_encoder_input_size=prompt_encoder_input_size,
        )
        box3D = torch.tensor(np.array([box2D[0], box2D[1], z_min, box2D[2], box2D[3], z_max]), dtype=torch.float32)

        segs_i = np.zeros(self.image_shape[:3], dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        box_default = np.array([x_min, y_min, x_max, y_max])
        z_middle = (z_max + z_min) // 2

        # infer from middle slice to the z_max
        box_2D = box_default
        for z in range(int(z_middle), int(z_max)):
            box_torch = torch.as_tensor(box_2D[None, ...], dtype=torch.float).to(self.device)  # (B, 4)
            mask, _ = self.model.prompt_and_decoder(self.embeddings[z], box_torch)
            mask = self.model.postprocess_masks(mask, new_size, self.original_size)
            mask = mask.squeeze().cpu().numpy()
            if np.max(mask) > 0:
                box_2D = get_bbox(mask)
                box_2D = resize_box(
                    box=box_2D,
                    original_size=self.original_size,
                    prompt_encoder_input_size=prompt_encoder_input_size,
                )
                segs_i[z, mask > 0] = 1
                res[z] = segs_i[z]
            else:
                box_2D = box_default

        # infer from middle slice to the z_min
        if np.max(segs_i[int(z_middle), :, :]) == 0:
            box_2D = box_default
        else:
            box_2D = get_bbox(segs_i[int(z_middle), :, :])
            box_2D = resize_box(
                box=box_2D,
                original_size=self.original_size,
                prompt_encoder_input_size=prompt_encoder_input_size,
            )

        for z in range(int(z_middle - 1), int(z_min - 1), -1):
            box_torch = torch.as_tensor(box_2D[None, ...], dtype=torch.float).to(self.device)  # (B, 4)
            mask, _ = self.model.prompt_and_decoder(self.embeddings[z], box_torch)
            mask = self.model.postprocess_masks(mask, new_size, self.original_size)
            mask = mask.squeeze().cpu().numpy()
            if np.max(mask) > 0:
                box_2D = get_bbox(mask)
                box_2D = resize_box(
                    box=box_2D,
                    original_size=self.original_size,
                    prompt_encoder_input_size=prompt_encoder_input_size,
                )
                segs_i[z, mask > 0] = 1
                res[z] = segs_i[z]
            else:
                box_2D = box_default

        return res


