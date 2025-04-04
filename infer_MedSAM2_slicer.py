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
import yaml

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def resize_rgb(array, image_size):
    d, h, w = array.shape[:3]
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_rgb = Image.fromarray(array[i].astype(np.uint8))
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def grayscale2rgb(array):
    if len(array.shape) > 3:
        return array

    rgb_array = np.zeros((*array.shape, 3))
    for i in range(array.shape[0]):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        rgb_array[i] = np.array(img_rgb)

    return rgb_array


@torch.inference_mode()
def infer_3d(predictor, img_npz_file, gts_file, propagate, model_cfg, pred_save_dir):
    print(f'infering {img_npz_file}')
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    gts = np.load(gts_file, 'r', allow_pickle=True)['segs'] if gts_file != 'X' else None
    img_3D = npz_data['imgs']  # (D, H, W)
    if np.max(img_3D) >= 256:
        img_3D = (img_3D - np.min(img_3D)) / (np.max(img_3D) - np.min(img_3D)) * 255
        img_3D = img_3D.astype(np.int16)
    # assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    img_3D = grayscale2rgb(img_3D)
    D, H, W = img_3D.shape[:3]
    segs_3D = np.zeros(img_3D.shape[:3], dtype=np.uint8)
    boxes_3D = npz_data['boxes']  # (D, num_boxes, 4)
    z_range = npz_data['z_range'] # (z_min, z_max, slice_idx)
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    with open(join('sam2', model_cfg), 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        image_size = yaml_data['model']['image_size']
    img_resized = resize_rgb(img_3D, image_size)
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
        img = img.astype(np.float32)
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
        ann_frame_idx = z_mid_orig - (z_min if z_min is not None else 0)

        print('analyzed image size', img.shape, 'mid idx', z_mid)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # input img is shape depth_to_consider, 3, 512, 512
            inference_state = predictor.init_state(img, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=ann_frame_idx, obj_id=1, mask=mask_prompt)
            segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            # run propagation throughout the video and collect the results in a dict
            #video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=ann_frame_idx, obj_id=1, mask=mask_prompt)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                print(out_frame_idx)
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)

    print(np.unique(segs_3D))
    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

    return inference_state


@torch.inference_mode()
def improve_3d(predictor, inference_state, img_npz_file, pred_save_dir):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    segs_3D = np.zeros(npz_data['img_size'], dtype=np.uint8)
    points_addition = npz_data['points_addition']
    points_subtraction = npz_data['points_subtraction']
    labels = np.array([1]*points_addition.shape[0] + [0]*points_subtraction.shape[0], dtype=np.int32)
    if points_addition.shape[0] == 0:
        points = points_subtraction
    elif points_subtraction.shape[0] == 0:
        points = points_addition
    else:
        points = np.vstack((points_addition, points_subtraction))
    print(labels, points, sep='\n')
    # pass points, labels, and the (box, zrange, slice_idx)
    box = np.array(npz_data['bboxes'][0], np.float32)
    z_min = min(npz_data['zrange'])
    z_mid_orig = int(points[0, -1])
    ann_frame_idx = z_mid_orig - z_min

    points = points[:,:-1].astype(np.float32) # dropping 3rd dimension

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = 1
        # run propagation throughout the video and collect the results in a dict
        #video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            print(out_frame_idx)
            segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            print(out_frame_idx)
            segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)

    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

    return inference_state



def perform_inference(checkpoint, cfg, img_path, gts_path, propagate, pred_save_dir):
    # make propagate boolean
    predictor = build_sam2_video_predictor_npz(cfg, checkpoint) if propagate else SAM2ImagePredictor(build_sam2(cfg, checkpoint, device="cuda"))

    os.makedirs(pred_save_dir, exist_ok=True)

    inference_state = infer_3d(predictor, img_path, gts_path, propagate, cfg, pred_save_dir)

    return predictor, inference_state

def improve_inference(img_path, pred_save_dir, predictor_state):
    # make propagate boolean
    predictor, inference_state = predictor_state['predictor'], predictor_state['inference_state']

    os.makedirs(pred_save_dir, exist_ok=True)

    inference_state = improve_3d(predictor, inference_state, img_path, pred_save_dir)

    return predictor, inference_state



if __name__ == '__main__':
    perform_inference('checkpoints/MedSAM2_latest.pt', 'MedSAM2_tiny512.yaml', 'img_data.npz', 'X', False, 'data/video/segs_tiny')
    print('Server is installed!')
