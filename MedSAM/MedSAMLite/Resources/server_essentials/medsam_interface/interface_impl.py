import sys
import time
import json
import os

from tqdm import tqdm

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from tiny_vit_sam import TinyViT
from PIL import Image
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer


##############################################################################################################################
####
####                                        vvvvvvvv  Main Code Area vvvvvvvv
####
##############################################################################################################################
def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw


def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    from scipy.ndimage import distance_transform_edt

    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im



def interp_shape(top, bottom, precision):
    from scipy.interpolate import interpn

    '''
    Interpolate between two contours

    Input: top
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")
    
    top, bottom = np.array(top), np.array(bottom)

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids
    points = (np.r_[0, 2], np.arange(r), np.arange(c))

    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
    xi = np.c_[np.full((r*c),precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out



class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks



class MedSAM_Interface:
    MedSAM_CKPT_PATH = None
    MEDSAM_IMG_INPUT_SIZE = 1024
    device = None

    H = None
    W = None
    image = None
    embeddings = None
    speed_level = 1

    def __init__(self):
        # 0. freeze seeds
        torch.manual_seed(2023)
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(2023)
        np.random.seed(2023)

        # settings and app states
        # self.MedSAM_CKPT_PATH = sys.argv[1] ################################################### FIXME
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embeddings = []
    
    @torch.no_grad()
    def medsam_inference(self, medsam_model, img_embed, box_1024, height, width):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
                low_res_pred,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg
    
    def build_medsam_lite(self):
        medsam_lite_image_encoder = TinyViT(
            img_size=256,
            in_chans=3,
            embed_dims=[
                64, ## (64, 256, 256)
                128, ## (128, 128, 128)
                160, ## (160, 64, 64)
                320 ## (320, 64, 64) 
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False, ## TODO: try MobileSAM's checkpoint next time
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )

        medsam_lite_prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16
        )

        medsam_lite_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
        )

        return MedSAM_Lite(
            image_encoder=medsam_lite_image_encoder,
            mask_decoder=medsam_lite_mask_decoder,
            prompt_encoder=medsam_lite_prompt_encoder
        )
    
    def load_model(self):
        # load MedSAM model
        print("Loading MedSAM lite model...")
        tic = time.perf_counter()
        # %% load model
        self.medsam_model = self.build_medsam_lite()
        medsam_checkpoint = torch.load(self.MedSAM_CKPT_PATH, map_location="cpu")
        self.medsam_model.load_state_dict(medsam_checkpoint)
        self.medsam_model.to(self.device)
        self.medsam_model.eval()
        print(f"MedSam lite loaded, took {time.perf_counter() - tic}")
    
    # receive number of slices, for each slice, receive the slice then calc embedding
    def set_image(self, arr, wmin, wmax, zmin, zmax, recurrent_func=None):
        # initialize params
        self.embeddings = []

        ###########################################################
        ### This should be handled during preprocessing
        ###########################################################
        # windowlization
        wmin = np.min(arr) - 1
        wmax = np.max(arr) + 1
        print('wmin: %g, wmax: %g'%(wmin, wmax))
        arr = np.clip(arr, wmin, wmax)
        arr = (arr - wmin) / (wmax - wmin) * 255

        # TODO: add restrictions on image dimension
        # assert (
            #     len(arr.shape) == 2 or arr.shape[-1] == 3
        # ), f"Accept either 1 channel gray image or 3 channel rgb. Got image shape {arr.shape} "
        self.image = arr
        self.H, self.W = arr.shape[1:]



        for slice_idx in range(self.image.shape[0]):
            if recurrent_func is not None:
                recurrent_func()
            slc = self.image[slice_idx]

            if len(slc.shape) == 2:
                img_3c = np.repeat(slc[:, :, None], 3, axis=-1)
            else:
                img_3c = slc

            img_1024 = transform.resize(
                    img_3c,
                (256, 256),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)

            # convert the shape to (3, H, W)
            img_1024_tensor = (
                    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            )
            
            mid_slice = (zmax + zmin) // 2
            calculation_condition = (zmax == -1) or ((zmin-1) <= slice_idx <= (zmax+1)) # Full embedding or partial embedding that lies between slices
            skip_condition = abs(slice_idx - mid_slice) % self.speed_level != 0
            if calculation_condition and not skip_condition:
                with torch.no_grad():
                    embedding = self.medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
            else:
                embedding = None

            self.embeddings.append(embedding)
    
    def get_bbox1024(self, mask_1024, bbox_shift=3):
        y_indices, x_indices = np.where(mask_1024 > 0)
        if x_indices.shape[0] == 0:
            return np.array([0, 0, bbox_shift, bbox_shift])
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        h, w = mask_1024.shape
        x_min = max(0, x_min - bbox_shift)
        x_max = min(w, x_max + bbox_shift)
        y_min = max(0, y_min - bbox_shift)
        y_max = min(h, y_max + bbox_shift)
        bboxes1024 = np.array([x_min, y_min, x_max, y_max])
        return bboxes1024

    def get_progress(self):
        return {'layers': 100 if self.image is None else self.image.shape[0], 'generated_embeds': len(self.embeddings)}

    def infer(self, 
        slice_idx: int,
        bbox: list[int],  # (xmin, ymin, xmax, ymax), origional size
        zrange: list[int], # (zmin, zmax), inference will be performed in this slice range, including zmin and zmax
    ):    
        print(slice_idx, bbox, zrange)
        zmin, zmax = zrange
        zmax = min(zmax+1, len(self.embeddings))
        zmin = max(zmin-1, 0)
        bbox_1024_prev = np.array([bbox]) / np.array([self.W, self.H, self.W, self.H]) * 256
        res = {}
        
        def inf(bbox_1024_prev):
            mask = self.medsam_inference(self.medsam_model, self.embeddings[csidx], bbox_1024_prev, self.H, self.W)
            mask_t = transform.resize(mask, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            return mask.tolist(), [self.get_bbox1024(mask_t)]

        # adjusting slice_idx
        available_idxs = np.array([i for i in range(len(self.embeddings)) if self.embeddings[i] is not None])
        print(available_idxs)
        slice_idx = available_idxs[np.argmin(np.abs(slice_idx - available_idxs))]
        print('Adjusted slice_idx: ', slice_idx)

        for csidx in range(slice_idx, zmax, self.speed_level):
            res[csidx], bbox_1024_prev = inf(bbox_1024_prev)
            if csidx == slice_idx:
                bbox_1024_center_inf = bbox_1024_prev.copy()

        bbox_1024_prev = bbox_1024_center_inf
        for csidx in range(slice_idx-self.speed_level, zmin, -(self.speed_level)):
            res[csidx], bbox_1024_prev = inf(bbox_1024_prev)
        

        if self.speed_level != 1: # Skip embeddings has happened
            frames = sorted(list(res.keys()))
            for idx in range(len(frames) - 1):
                frames_distance = frames[idx+1] - frames[idx]  # it is == speed_level except for middle slice
                for f_idx in range(1, frames_distance):
                    intermediate_slice = interp_shape(res[frames[idx+1]], res[frames[idx]], f_idx/frames_distance)
                    res[frames[idx]+f_idx] = intermediate_slice

        return res

