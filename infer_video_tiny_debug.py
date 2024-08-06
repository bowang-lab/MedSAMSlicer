import torch
from sam2.build_sam import build_sam2
from glob import glob
from tqdm import tqdm
from time import time
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
from os.path import join, isfile, basename
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2
import cv2
import SimpleITK as sitk
import random
import argparse

# %%
parser = argparse.ArgumentParser(description="Medical SAM2 inference")
parser.add_argument("--img_path", type=str, help="Path to the input image")
parser.add_argument("--gts_path", type=str, help="Path to the ground truth. Use 'X' for null")
parser.add_argument("--propagate", type=str, help="y/n whether to propagate from the middle slice")
parser.add_argument("--checkpoint", type=str, help="checkpoint size")
args = parser.parse_args()
# %%
img_path = args.img_path
gts_path = args.gts_path
propagate = args.propagate in ['y', 'Y']
model_checkpoint = args.checkpoint

random.seed(42)
 

# tiny
checkpoint = "./checkpoints/sam2_hiera_%s.pt"%(model_checkpoint,)
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint) if propagate else SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cuda"))
save_overlay=True
png_save_dir = 'data/video/overlay_tiny'
os.makedirs(png_save_dir, exist_ok=True)
gt_path = 'data/gts'
data_root = 'data/imgs'
pred_save_dir = 'data/video/segs_tiny'
os.makedirs(pred_save_dir, exist_ok=True)
#nifti_path = '/home/sumin2/Documents/segment-anything-2/data/video/segs_tiny_nifti'
#os.makedirs(nifti_path, exist_ok=True)


def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

def resize_bicubic_cv2(array, new_shape):
    """
    Resize a 3D NumPy array using bicubic interpolation.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        new_shape (tuple): Desired shape (new_d, new_h, new_w).
    
    Returns:
        np.ndarray: Resized array of shape (new_d, new_h, new_w).
    """
    d, h, w = array.shape
    new_d, new_h, new_w = new_shape
    resized_array = np.zeros((new_d, new_h, new_w))
    
    for i in range(d):
        resized_array[i] = cv2.resize(array[i], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return resized_array




@torch.inference_mode()
def infer_3d(img_npz_file, gts_file, propagate):
    print(f'infering {img_npz_file}')
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    gts = np.load(gts_file, 'r', allow_pickle=True)['segs'] if gts_file != 'X' else None
    img_3D = npz_data['imgs']  # (D, H, W)
    assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    D, H, W = img_3D.shape
    segs_3D = np.zeros(img_3D.shape, dtype=np.uint8)
    boxes_3D = npz_data['boxes']  # (D, num_boxes, 4)
    z_range = npz_data['z_range'] # (z_min, z_max, slice_idx)
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    img_resized = resize_bicubic_cv2(img_3D, (D, 1024, 1024))
    img_resized = img_resized / 255.0
    img_resized = np.repeat(img_resized[:, None], 3, axis=1) # d, 3, 512, 512
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
                print('begin forward propagation')
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                print('begin reverse propagation')
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)

    print(np.unique(segs_3D))
    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)


if __name__ == '__main__':
    # img_npz_files = sorted(glob(join(data_root, '3D*.npz'), recursive=True))
    img_npz_files = [img_path]
    gts_files = [gts_path] 
    
    print(img_npz_files)
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file, gts_file in tqdm(zip(img_npz_files, gts_files)):
        start_time = time()
        #if basename(img_npz_file).startswith('3D'):
        infer_3d(img_npz_file, gts_file, propagate)
        #else:
        #    infer(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
