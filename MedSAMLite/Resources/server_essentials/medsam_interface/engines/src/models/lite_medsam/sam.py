import torch

from src.models.segment_anything.modeling import (
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)

from .tiny_vit import TinyViT
from ..base_sam import BaseSAM


def build_lite_medsam(checkpoint=None):
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64,  ## (64, 256, 256)
            128,  ## (128, 128, 128)
            160,  ## (160, 64, 64)
            320,  ## (320, 64, 64)
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
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

    sam = BaseSAM(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
