from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segment_anything.modeling import MaskDecoder, PromptEncoder


class BaseSAM(nn.Module):

    def __init__(
        self,
        image_encoder: nn.Module,
        mask_decoder: MaskDecoder,
        prompt_encoder: PromptEncoder,
        multimask_output: bool = False,
        return_best_mask: bool = True,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.multimask_output = multimask_output
        self.return_best_mask = return_best_mask

    def prompt_and_decoder(
        self, image_embedding: torch.Tensor, boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        # I: number of image embeddings
        # B: number of boxes
        # Assume that each image has the same number of boxes (= B / I)
        # M: number of multimask outputs (default is 3 if multimask_output is True, otherwise 1)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (I, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=self.multimask_output,
        )  # (B, M, 256, 256)

        if self.multimask_output and self.return_best_mask:
            max_values, max_indices = torch.max(iou_predictions, dim=1)
            iou_predictions = max_values.unsqueeze(1)
            low_res_masks = torch.take_along_dim(
                low_res_masks, indices=max_indices.view(-1, 1, 1, 1), dim=1
            )  # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    def forward(self, image: torch.Tensor, boxes: torch.Tensor):
        image_embedding = self.image_encoder(image)
        return self.prompt_and_decoder(image_embedding, boxes)

    @torch.no_grad()
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
        return_with_image_encoder_size: bool = False,
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (max(input_size), max(input_size)),
            mode="bilinear",
            align_corners=False,
        )
        if return_with_image_encoder_size:
            return masks

        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks,
            original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

    @classmethod
    def construct_from(
        cls,
        original_sam: Optional[nn.Module] = None,
        distill_lit_module=None,
        finetune_lit_module=None,
        multimask_output: bool = False,
        return_best_mask: bool = True,
    ):
        if finetune_lit_module is not None:
            if isinstance(finetune_lit_module.model, cls):
                return finetune_lit_module.model
            original_sam = finetune_lit_module.model

        assert original_sam is not None
        image_encoder = original_sam.image_encoder
        mask_decoder = original_sam.mask_decoder
        prompt_encoder = original_sam.prompt_encoder
        if distill_lit_module is not None:
            image_encoder = distill_lit_module.student_encoder
        return cls(
            image_encoder=image_encoder,
            mask_decoder=mask_decoder,
            prompt_encoder=prompt_encoder,
            multimask_output=multimask_output,
            return_best_mask=return_best_mask,
        )
