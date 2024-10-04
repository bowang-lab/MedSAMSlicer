from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.nn import functional as F


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
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class MinMaxScale(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image should have shape (..., 3, H, W)
        """
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


def get_bbox(mask: np.ndarray, bbox_shift: int = 0) -> np.ndarray:
    """
    Get the bounding box coordinates from the mask

    Parameters
    ----------
    mask : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W - 1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H - 1, y_max + bbox_shift)

    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


def resize_box(
    box: np.ndarray,
    original_size: Tuple[int, int],
    prompt_encoder_input_size: int,
) -> np.ndarray:
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image
    prompt_encoder_input_size : int
        the target size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
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


def transform_gt(gt: torch.Tensor, long_side_length: int):
    gt = gt[None, None, ...]
    oldh, oldw = gt.shape[-2:]
    newh, neww = ResizeLongestSide.get_preprocess_shape(oldh, oldw, long_side_length)
    gt = F.interpolate(gt, (newh, neww), mode="nearest-exact")
    gt = F.pad(gt, (0, long_side_length - neww, 0, long_side_length - newh), value=0)
    return gt.squeeze((0, 1))
