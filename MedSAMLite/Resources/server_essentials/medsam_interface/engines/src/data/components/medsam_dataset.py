import itertools
import os
import random
import zipfile
from glob import glob
from time import time
from typing import List, Optional

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.multiprocessing import parmap
from src.utils.transforms import (
    ResizeLongestSide,
    get_bbox,
    get_image_transform,
    resize_box,
    transform_gt,
)


class MedSAMBaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_encoder_input_size: int = 512,
        prompt_encoder_input_size: Optional[int] = None,
        scale_image: bool = True,
        normalize_image: bool = False,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        interpolation: str = "bilinear",
    ):
        self.data_dir = data_dir
        self.image_encoder_input_size = image_encoder_input_size
        self.prompt_encoder_input_size = (
            prompt_encoder_input_size
            if prompt_encoder_input_size is not None
            else image_encoder_input_size
        )
        self.scale_image = scale_image
        self.normalize_image = normalize_image
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.interpolation = interpolation
        self.transform_image = get_image_transform(
            long_side_length=self.image_encoder_input_size,
            min_max_scale=self.scale_image,
            normalize=self.normalize_image,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            interpolation=self.interpolation,
        )


class MedSAMTrainDataset(MedSAMBaseDataset):
    def __init__(
        self,
        bbox_random_shift: int = 5,
        mask_num: int = 5,
        data_aug: bool = True,
        num_workers: int = 8,
        glob_pattern: str = "**/*.npz",
        limit_npz: Optional[int] = None,
        limit_sample: Optional[int] = None,
        aug_transform: Optional[A.TransformType] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.bbox_random_shift = bbox_random_shift
        self.mask_num = mask_num

        self.npz_file_paths = sorted(
            glob(os.path.join(self.data_dir, glob_pattern), recursive=True)
        )
        if limit_npz is not None:
            self.npz_file_paths = self.npz_file_paths[:limit_npz]

        self.items = list(
            itertools.chain.from_iterable(
                parmap(self.__flatten_npz, self.npz_file_paths, nprocs=num_workers)
            )
        )
        if limit_sample is not None:
            rng = random.Random(42)
            self.items = rng.sample(self.items, limit_sample)

        print("Number of samples:", len(self.items))

        if not data_aug:
            self.aug_transform = A.NoOp()
        elif aug_transform is not None:
            self.aug_transform = aug_transform
        else:
            self.aug_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )

    def __flatten_npz(self, npz_file_path):
        try:
            data = np.load(npz_file_path, "r")
        except zipfile.BadZipFile:
            return []

        gts = data["gts"]
        assert len(gts.shape) == 2 or len(gts.shape) == 3
        if len(gts.shape) > 2:  # 3D
            return [
                (npz_file_path, slice_index)
                for slice_index in gts.max(axis=(1, 2)).nonzero()[0]
            ]
        else:  # 2D
            return [(npz_file_path, -1)] if gts.max() > 0 else []

    def get_name(self, item):
        name = os.path.basename(item[0]).split(".")[0]
        return name + f"_{item[1]:03d}" if item[1] != -1 else name

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        data = np.load(item[0], "r")
        img = data["imgs"]
        gt = data["gts"]  # multiple labels [0, 1, 4, 5, ...], (H, W)

        if item[1] != -1:  # 3D
            img = img[item[1], :, :]
            gt = gt[item[1], :, :]

        # duplicate channel if the image is grayscale
        if len(img.shape) < 3:
            img = np.repeat(img[..., None], 3, axis=-1)  # (H, W, 3)

        labels = np.unique(gt[gt > 0])
        assert len(labels) > 0, f"No label found in {item[0]}"
        labels = random.choices(labels, k=self.mask_num)

        # augmentation
        all_masks = [np.array(gt == label, dtype=np.uint8) for label in labels]
        augmented = self.aug_transform(image=img, masks=all_masks)
        img, all_masks = augmented["image"], augmented["masks"]
        original_size = img.shape[:2]

        # Extract boxes and masks from ground-truths
        masks_list = []
        boxes_list = []
        for mask in all_masks:
            mask = torch.from_numpy(mask).type(torch.uint8)
            mask = transform_gt(mask, self.image_encoder_input_size)
            if mask.max() == 0:
                H, W = mask.shape
                x_min = random.randint(0, W - 1)
                x_max = random.randint(0, W - 1)
                y_min = random.randint(0, H - 1)
                y_max = random.randint(0, H - 1)
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                if y_min > y_max:
                    y_min, y_max = y_max, y_min

                bbox_shift = 1
                x_min = max(0, x_min - bbox_shift)
                x_max = min(W - 1, x_max + bbox_shift)
                y_min = max(0, y_min - bbox_shift)
                y_max = min(H - 1, y_max + bbox_shift)

                box = np.array([x_min, y_min, x_max, y_max])
            else:
                box = get_bbox(mask, random.randint(0, self.bbox_random_shift))
            box = resize_box(box, mask.shape, self.prompt_encoder_input_size)
            box = torch.tensor(box, dtype=torch.float32)
            masks_list.append(mask)
            boxes_list.append(box)

        tsfm_img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.uint8)
        tsfm_img = self.transform_image(tsfm_img.unsqueeze(0)).squeeze(0)

        return {
            "image": tsfm_img,  # (3, H, W)
            "masks": torch.stack(masks_list).unsqueeze(1),  # (N, H, W)
            "boxes": torch.stack(boxes_list),  # (N, 4)
            "original_size": torch.tensor(original_size, dtype=torch.int32),
        }


class MedSAMDistillDataset(MedSAMTrainDataset):
    def __init__(
        self,
        teacher_image_encoder_input_size: Optional[int] = 1024,
        teacher_scale_image: bool = True,
        teacher_normalize_image: bool = False,
        embedding_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.teacher_image_encoder_input_size = teacher_image_encoder_input_size
        self.teacher_scale_image = teacher_scale_image
        self.teacher_normalize_image = teacher_normalize_image
        if teacher_image_encoder_input_size is not None:
            self.transform_teacher_image = get_image_transform(
                long_side_length=self.teacher_image_encoder_input_size,
                min_max_scale=self.teacher_scale_image,
                normalize=self.teacher_normalize_image,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
                interpolation=self.interpolation,
            )

        self.embedding_dir = embedding_dir
        if self.embedding_dir is not None:
            self.items = self.__filter_valid_embs(self.items, embedding_dir)

    def __filter_valid_embs(self, items, embedding_dir):
        """
        Filter the npz_file_paths, ignore file that does not have image embedding
        Some embedding maybe missed during feature extraction process
        """

        valid = []
        for item in items:
            name = self.get_name(item)
            npy_file_path = os.path.join(embedding_dir, name + ".npy")
            if os.path.exists(npy_file_path):
                valid.append(item)
        print(f"Found {len(valid)} image embeddings.")
        return valid

    def __getitem__(self, index):
        item = self.items[index]
        data = np.load(item[0], "r")
        img = data["imgs"]

        if item[1] != -1:  # 3D
            img = img[item[1], :, :]

        # duplicate channel if the image is grayscale
        if len(img.shape) < 3:
            img = np.repeat(img[..., None], 3, axis=-1)  # (H, W, 3)

        # augmentation
        tsfm_img = self.aug_transform(image=img)["image"]
        tsfm_img = torch.tensor(np.transpose(tsfm_img, (2, 0, 1)), dtype=torch.uint8)

        items = {"image": self.transform_image(tsfm_img.unsqueeze(0)).squeeze(0)}

        if self.teacher_image_encoder_input_size is not None:
            # Transform image
            items["teacher_image"] = self.transform_teacher_image(
                tsfm_img.unsqueeze(0)
            ).squeeze(0)
        elif self.embedding_dir is not None:
            img_name = self.get_name(item)
            emb_file = os.path.join(self.embedding_dir, img_name + ".npy")
            items["embedding"] = np.load(emb_file, "r", allow_pickle=True)

        return items


class MedSAMInferDataset(MedSAMBaseDataset):
    def __init__(self, glob_pattern: str = "**/*.npz", **kwargs):
        super().__init__(**kwargs)
        self.npz_file_paths = sorted(
            glob(os.path.join(self.data_dir, glob_pattern), recursive=True)
        )

    def __len__(self):
        return len(self.npz_file_paths)

    def __getitem__(self, index):
        start_time = time()

        npz_file_path = self.npz_file_paths[index]
        npz_name = os.path.basename(npz_file_path)
        data = np.load(npz_file_path, "r")
        img = data["imgs"]
        boxes = data["boxes"]

        if os.path.basename(npz_file_path).startswith("2D"):
            if len(img.shape) < 3:
                img = np.repeat(img[..., None], 3, axis=-1)  # (H, W, 3)

            original_size = img.shape[:2]
            new_size = ResizeLongestSide.get_preprocess_shape(
                original_size[0], original_size[1], self.image_encoder_input_size
            )
            tsfm_img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.uint8)
            tsfm_img = self.transform_image(tsfm_img.unsqueeze(0)).squeeze(0)

            # Transform box
            tsfm_boxes = []
            for box in boxes:
                box = resize_box(
                    box,
                    original_size=original_size,
                    prompt_encoder_input_size=self.prompt_encoder_input_size,
                )
                tsfm_boxes.append(box)

            end_time = time()
            print(f"Processed {npz_name} in {end_time - start_time:.2f}s")

            return {
                "image": tsfm_img,  # (3, H, W)
                "boxes": torch.tensor(
                    np.array(tsfm_boxes), dtype=torch.float32
                ),  # (N, 4)
                "npz_name": npz_name,
                "new_size": torch.tensor(new_size, dtype=torch.int32),
                "original_size": torch.tensor(original_size, dtype=torch.int32),
                "image_type": "2D",
                "original_image": img,
                "original_boxes": boxes,
            }

        elif os.path.basename(npz_file_path).startswith("3D"):
            if len(img.shape) == 3:
                # gray: (D, H, W) -> (D, 3, H, W)
                tsfm_imgs = np.repeat(img[:, None, ...], 3, axis=1)
            else:
                # rbg: (D, H, W, 3) -> (D, 3, H, W)
                tsfm_imgs = np.transpose(img, (0, 3, 1, 2))

            original_size = img.shape[-2:]
            new_size = ResizeLongestSide.get_preprocess_shape(
                original_size[0], original_size[1], self.image_encoder_input_size
            )
            tsfm_imgs = self.transform_image(torch.tensor(tsfm_imgs, dtype=torch.uint8))

            # Transform box
            tsfm_boxes = []
            for box3D in boxes:
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box2D = np.array([x_min, y_min, x_max, y_max])
                box2D = resize_box(
                    box2D,
                    original_size=original_size,
                    prompt_encoder_input_size=self.prompt_encoder_input_size,
                )
                box3D = np.array([box2D[0], box2D[1], z_min, box2D[2], box2D[3], z_max])
                tsfm_boxes.append(box3D)

            end_time = time()
            print(f"Processed {npz_name} in {end_time - start_time:.2f}s")

            return {
                "image": tsfm_imgs,  # (D, 3, H, W)
                "boxes": torch.tensor(
                    np.array(tsfm_boxes), dtype=torch.float32
                ),  # (N, 6)
                "npz_name": npz_name,
                "new_size": torch.tensor(new_size, dtype=torch.int32),
                "original_size": torch.tensor(original_size, dtype=torch.int32),
                "prompt_encoder_input_size": self.prompt_encoder_input_size,
                "image_type": "3D",
                "original_image": img,
                "original_boxes": boxes,
            }

        raise Exception(
            f"Unexpected input type for file {npz_file_path}, only allow 3D- and 2D- prefix"
        )
