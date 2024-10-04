import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segment_anything.modeling import MaskDecoder, PromptEncoder


class DecoderOnnxModel(nn.Module):
    def __init__(
        self,
        mask_decoder: MaskDecoder,
        prompt_encoder: PromptEncoder,
        image_encoder_input_size: int = 512,
    ):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.image_encoder_input_size = image_encoder_input_size

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        boxes: torch.Tensor,
    ):
        coords = boxes.reshape(-1, 2, 2)
        sparse_embeddings = self.prompt_encoder.pe_layer._pe_encoding(coords)
        sparse_embeddings[:, 0, :] += self.prompt_encoder.point_embeddings[2].weight
        sparse_embeddings[:, 1, :] += self.prompt_encoder.point_embeddings[3].weight

        dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand(
            1,
            -1,
            self.prompt_encoder.image_embedding_size[0],
            self.prompt_encoder.image_embedding_size[1],
        )

        masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            masks,
            (self.image_encoder_input_size, self.image_encoder_input_size),
            mode="bilinear",
            align_corners=False,
        )

        return masks
