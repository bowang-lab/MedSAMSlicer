### Code and models are adopted from https://github.com/automl/CVPR24-MedSAM-on-Laptop

from os import makedirs
from os.path import join, basename, dirname
from glob import glob
import numpy as np

import cv2
import openvino as ov
import openvino.properties as props


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

    if image.dtype == np.int32: # it can cause opencv assertion error
        image = image.astype(np.int16)

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


class DAFTSAMCore:
    model = None
    device = None
    MedSAM_CKPT_PATH = None
    device = None

    H = None
    W = None
    D = None
    image_shape = None
    embeddings = None

    def __init__(self):
        self.core = ov.Core()

    def load_model(self):
        self.model_root = dirname(self.MedSAM_CKPT_PATH)
        self.pos_encoding = np.load(join(dirname(self.model_root), "positional_encoding.npy"))
        self.pe = self.core.compile_model(model=join(dirname(self.model_root), "prompt_encoder.xml"), device_name="CPU")
        self.sessions = dict()
        self.image_encoder, self.prompt_encoder, self.mask_decoder, self.positional_encoding = self.load_session(basename(self.model_root)) #FIXME

    def get_progress(self):
        return {'layers': 100 if self.image_shape is None else self.image_shape[0], 'generated_embeds': len(self.embeddings)}
    
    def load_session(self, name):
        if name not in self.sessions:
            ie = self.core.compile_model(model=join(self.model_root, "image_encoder.xml"), device_name="CPU")
            md = self.core.compile_model(model=join(self.model_root, "mask_decoder.xml"), device_name="CPU")
            self.sessions[name] = [ie, self.pe, md, self.pos_encoding]
        return self.sessions[name]


    def set_image(self, image_data, wmin, wmax, zmin, zmax, recurrent_func):
        self.embeddings = []

        self.image_shape = image_data.shape
        self.original_size = image_data.shape[1:3]
        self.H, self.W = self.original_size

        def compute_embedding(img_2d):
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]
            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            # convert the shape to (3, H, W)
            img_256 = img_256.astype(np.float32).transpose((2, 0, 1))[None, ...]
            # get the image embedding
            image_embedding = self.image_encoder({"image":img_256})[0]
            return  image_embedding, new_H, new_W

        segs = np.zeros_like(image_data, dtype=np.uint16)

        self.new_H, self.new_W = None, None
        
        for z in range(image_data.shape[0]):
            if recurrent_func is not None:
                recurrent_func()
            image_embedding = None
            calculation_condition = (zmax == -1) or ((zmin-1) <= z <= (zmax+1)) # Full embedding or partial embedding that lies between slices
            if calculation_condition:
                img_2d = image_data[z, :, :]
                image_embedding, self.new_H, self.new_W = compute_embedding(img_2d)  # (1, 256, 64, 64)
            else:
                image_embedding = None
            self.embeddings.append(image_embedding)
    
    
    def medsam_inference(self, prompt_encoder, mask_decoder, positional_encoding, img_embed, box_256, new_size, original_size):
        """
        Perform inference using the LiteMedSAM model.

        Args:
            medsam_model (MedSAMModel): The MedSAM model.
            img_embed (torch.Tensor): The image embeddings.
            box_256 (numpy.ndarray): The bounding box coordinates.
            new_size (tuple): The new size of the image.
            original_size (tuple): The original size of the image.
        Returns:
            tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
        """
        out = prompt_encoder({"boxes":box_256[None, None, ...].astype(np.float32)})
        sparse_embeddings, dense_embeddings = out["sparse_embeddings"], out["dense_embeddings"]
        out = mask_decoder({"image_embeddings":img_embed, "image_pe": positional_encoding, "sparse_prompt_embeddings": sparse_embeddings, "dense_prompt_embeddings": dense_embeddings})
        low_res_logits, iou = out["low_res_masks"], out["iou_predictions"]
        low_res_logits = low_res_logits[..., :new_size[0], :new_size[1]]
        # Resize
        low_res_logits = low_res_logits.squeeze()
        low_res_logits = cv2.resize(low_res_logits, original_size[::-1], interpolation=cv2.INTER_LINEAR)
        medsam_seg = (low_res_logits > 0).astype(np.uint8)
        return medsam_seg, iou


    def infer(self, slice_idx, bbox, zrange):
        z_min, z_max = zrange
        z_max = min(z_max+1, len(self.embeddings))
        z_min = max(z_min-1, 0)
        x_min, y_min, x_max, y_max = bbox

        segs_3d_temp = np.zeros(self.image_shape[:3], dtype=np.uint16)
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        res = {}

        for z in range(z_middle, z_max):
            image_embedding=self.embeddings[z]
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(self.H, self.W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    if np.max(pre_seg256) > 0:
                        box_256 = get_bbox256(pre_seg256)
                    else:
                        box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(self.H, self.W))
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(self.H, self.W))
            img_2d_seg, iou_pred = self.medsam_inference(self.prompt_encoder, self.mask_decoder, self.positional_encoding, image_embedding, box_256[None, ...], [self.new_H, self.new_W], [self.H, self.W])
            segs_3d_temp[z, img_2d_seg>0] = 1
            res[z] = segs_3d_temp[z]

        z_min = max(-1, z_min-1)
        for z in range(z_middle-1, z_min, -1):
            image_embedding=self.embeddings[z]
            pre_seg = segs_3d_temp[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                if np.max(pre_seg256) > 0:
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(self.H, self.W))
            else:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(self.H, self.W))
            img_2d_seg, iou_pred = self.medsam_inference(self.prompt_encoder, self.mask_decoder, self.positional_encoding, image_embedding, box_256[None, ...], [self.new_H, self.new_W], [self.H, self.W])
            segs_3d_temp[z, img_2d_seg>0] = 1
            res[z] = segs_3d_temp[z]

        return res


