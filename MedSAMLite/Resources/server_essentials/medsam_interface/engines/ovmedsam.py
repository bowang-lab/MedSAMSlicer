import sys
import time
import json
import os

from tqdm import tqdm

import numpy as np
from skimage import transform, io
import openvino as ov
from PIL import Image
import cv2
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer


##############################################################################################################################
####
####                                        vvvvvvvv  Main Code Area vvvvvvvv
####
##############################################################################################################################
### next four functions were adopted from https://github.com/Zrrr1997/medsam_cvhci
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



class ov_MedSAM_Lite():
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder,
            positional_encoding,
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.positional_encoding = positional_encoding

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



class OVMedSAMCore:
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
        # torch.manual_seed(2023)
        # torch.cuda.empty_cache()
        # torch.cuda.manual_seed(2023)
        np.random.seed(2023)

        # settings and app states
        # self.MedSAM_CKPT_PATH = sys.argv[1] ################################################### FIXME
        self.device = 'CPU' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embeddings = []
    
    def medsam_inference(self, medsam_model, img_embed, box_1024, height, width):
        if len(box_1024.shape) == 2:
            box_1024 = box_1024[:, None, :]  # (B, 1, 4)

        try:
            print('Trying one None')
            out = medsam_model.prompt_encoder({"boxes":box_1024[None, ...].astype(np.float32)})
        except Exception as error:
            print('One Failed with err', error)
            try:
                print('Trying two Nones')
                out = medsam_model.prompt_encoder({"boxes":box_1024[None, None, ...].astype(np.float32)})
            except Exception as error:
                print('One Failed with err', error)
                try:
                    print('Trying three Nones')
                    out = medsam_model.prompt_encoder({"boxes":box_1024[None, None, None, ...].astype(np.float32)})
                except Exception as error:
                    print('Three Failed with err', error)
                    try:
                        print('Trying no Nones')
                        out = medsam_model.prompt_encoder({"boxes":box_1024.astype(np.float32)})
                    except Exception as error:
                        print('Zero Failed with err', error)
                        
                    
                
        sparse_embeddings, dense_embeddings = out["sparse_embeddings"], out["dense_embeddings"]
        out = medsam_model.mask_decoder({
            "image_embeddings": img_embed,  # (B, 256, 64, 64)
            "image_pe": medsam_model.positional_encoding,  # (1, 256, 64, 64)
            "sparse_prompt_embeddings": sparse_embeddings,  # (B, 2, 256)
            "dense_prompt_embeddings": dense_embeddings,  # (B, 256, 64, 64)
        })
        low_res_logits = out["low_res_masks"]

        low_res_logits, iou = out["low_res_masks"], out["iou_predictions"]
        # low_res_logits = low_res_logits[..., :new_size[0], :new_size[1]]
        # Resize
        low_res_logits = low_res_logits.squeeze()
        low_res_logits = cv2.resize(low_res_logits, (width, height), interpolation=cv2.INTER_LINEAR)
        medsam_seg = (low_res_logits > 0).astype(np.uint8)
        
        return medsam_seg

    
    def build_medsam_lite(self):
        core = ov.Core()

        pos_encoding = np.load(os.path.join(os.path.dirname(self.MedSAM_CKPT_PATH), "positional_encoding.npy"))
        pe = core.compile_model(model=os.path.join(os.path.dirname(self.MedSAM_CKPT_PATH), "medsam_lite_prompt_encoder.xml"), device_name="CPU")
        ie = core.compile_model(model=os.path.join(os.path.dirname(self.MedSAM_CKPT_PATH), "medsam_lite_image_encoder.xml"), device_name="CPU")
        md = core.compile_model(model=os.path.join(os.path.dirname(self.MedSAM_CKPT_PATH), "medsam_lite_mask_decoder.xml"), device_name="CPU")

        return ov_MedSAM_Lite(
            image_encoder=ie,
            mask_decoder=md,
            prompt_encoder=pe,
            positional_encoding=pos_encoding
        )
    
    def load_model(self):
        # load MedSAM model
        print("Loading MedSAM lite model...")
        tic = time.perf_counter()
        # %% load model
        self.medsam_model = self.build_medsam_lite()
        # medsam_checkpoint = torch.load(self.MedSAM_CKPT_PATH, map_location="cpu") ############# FIXME
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
        self.H, self.W = arr.shape[1:3]



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
            img_1024_tensor = img_1024.astype(np.float32).transpose((2, 0, 1))[None, ...]
            
            mid_slice = (zmax + zmin) // 2
            calculation_condition = (zmax == -1) or ((zmin-1) <= slice_idx <= (zmax+1)) # Full embedding or partial embedding that lies between slices
            skip_condition = abs(slice_idx - mid_slice) % self.speed_level != 0
            if calculation_condition and not skip_condition:
                embedding = self.medsam_model.image_encoder({"image":img_1024_tensor})[0]  # (1, 256, 64, 64)
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
            print('==========================', csidx, bbox_1024_prev, '==========================')
            mask = self.medsam_inference(self.medsam_model, self.embeddings[csidx], bbox_1024_prev[None, ...], self.H, self.W)
            mask_t = transform.resize(mask, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            return mask.tolist(), self.get_bbox1024(mask_t).reshape(1, -1)

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

