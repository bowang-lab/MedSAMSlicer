from glob import glob
from tqdm import tqdm
import os
from os.path import join, isfile, basename
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime
from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="checkpoints/2.1/sam2.1_hiera_tiny.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="configs/sam2.1/sam2.1_hiera_t.yaml",
    help='model config',
)

parser.add_argument("--img_path", type=str, help="Path to the input image")
parser.add_argument("--gts_path", type=str, help="Path to the ground truth. Use 'X' for null")
parser.add_argument("--propagate", type=str, help="y/n whether to propagate from the middle slice")

parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="seg_results/segs_tiny",
    help='segs path',
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
img_path = args.img_path
gts_path = args.gts_path
propagate = args.propagate in ['y', 'Y']
pred_save_dir = args.pred_save_dir

predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint) if propagate else SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cuda"))

os.makedirs(pred_save_dir, exist_ok=True)


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array


@torch.inference_mode()
def infer_3d(img_npz_file, gts_file, propagate):
    print(f'infering {img_npz_file}')
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    gts = np.load(gts_file, 'r', allow_pickle=True)['segs'] if gts_file != 'X' else None
    img_3D = npz_data['imgs']  # (D, H, W)
    if np.max(img_3D) >= 256:
        img_3D = (img_3D - np.min(img_3D)) / (np.max(img_3D) - np.min(img_3D)) * 255
        img_3D = img_3D.astype(np.int16)
    # assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    D, H, W = img_3D.shape
    segs_3D = np.zeros(img_3D.shape, dtype=np.uint8)
    boxes_3D = npz_data['boxes']  # (D, num_boxes, 4)
    z_range = npz_data['z_range'] # (z_min, z_max, slice_idx)
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    img_resized = resize_grayscale_to_rgb_and_resize(img_3D, 1024)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    z_mids = []
    
    z_indices, slice_idx = z_range[:2], z_range[2]

    if not propagate:
        # predicting only middle slice
        img = img_3D[slice_idx] / 255.
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2).astype(np.float32)
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes_3D, multimask_output=False,)
        if len(masks.shape) == 3: # single bounding box
            masks = [masks]
        for idx, mask in enumerate(masks, start=1):
            segs_3D[slice_idx, (mask[0] > 0.0)] = idx
        np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

        print('Middle Slice Mask Calculated')

        return


    for idx, points in enumerate(boxes_3D, start=1):
        gt = (gts == (idx))
        indices = np.where(gt)
        z_mid_orig = indices[0][0]

        z_min = z_indices.min() if z_indices.size > 0 else None
        z_max = z_indices.max() if z_indices.size > 0 else None

        img = img_resized[z_min:(z_max+1)]
        z_mid = int(img.shape[0]/2)
        z_mids.append(z_mid_orig)
        mask_prompt = gt[z_mid_orig]

        print('analyzed image size', img.shape, 'mid idx', z_mid)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # input img is shape depth_to_consider, 3, 512, 512
            inference_state = predictor.init_state(img, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            # run propagation throughout the video and collect the results in a dict
            #video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)

    print(np.unique(segs_3D))
    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

if __name__ == '__main__':
    infer_3d(img_path, gts_path, propagate)

