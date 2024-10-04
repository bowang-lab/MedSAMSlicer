import io
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import torch
import torch.nn as nn
import hydra
import rootutils
import onnxruntime as ort
import openvino as ov
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from onnxsim import simplify
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.base_sam import BaseSAM
from src.models.onnx import EncoderOnnxModel, DecoderOnnxModel
from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def convert_to_openvino(onnx_model: Path):
    model = ov.convert_model(onnx_model)
    ov.save_model(model, onnx_model.with_suffix(".xml"), compress_to_fp16=False)


def export_onnx(
    cfg: DictConfig,
    onnx_model: nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]],
    output_names: List[str],
    output_file: Path,
):
    _ = onnx_model(**dummy_inputs)

    buffer = io.BytesIO()
    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        buffer,
        export_params=True,
        verbose=False,
        opset_version=cfg.opset,
        do_constant_folding=True,
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    buffer.seek(0, 0)

    if cfg.simplify:
        onnx_model = onnx.load_model(buffer)
        onnx_model, success = simplify(onnx_model)
        assert success
        new_buffer = io.BytesIO()
        onnx.save(onnx_model, new_buffer)
        buffer = new_buffer
        buffer.seek(0, 0)

    with open(output_file, "wb") as f:
        f.write(buffer.read())

    optimized_output_file = output_file.with_suffix(".optimized.onnx")
    opt = ort.SessionOptions()
    opt.optimized_model_filepath = optimized_output_file.as_posix()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ = ort.InferenceSession(output_file, opt, providers=["CPUExecutionProvider"])

    quantized_output_file = output_file.with_suffix(".quantized.onnx")
    quantize_dynamic(
        model_input=output_file,
        model_output=quantized_output_file,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )

    convert_to_openvino(output_file)


def export_encoder(sam_model: BaseSAM, cfg: DictConfig):
    onnx_model = EncoderOnnxModel(
        image_encoder=sam_model.image_encoder,
        preprocess_image=cfg.encoder_config.preprocess_image,
        image_encoder_input_size=cfg.encoder_config.image_encoder_input_size,
        scale_image=cfg.encoder_config.scale_image,
        normalize_image=cfg.encoder_config.normalize_image,
    )

    if cfg.encoder_config.preprocess_image:
        dummy_inputs = {
            "image": torch.randint(0, 256, (256, 384, 3), dtype=torch.uint8),
            "original_size": torch.tensor([256, 384], dtype=torch.int16),
        }
        dynamic_axes = {"image": {0: "image_height", 1: "image_width"}}
        output_names = ["image_embeddings"]
    else:
        dummy_inputs = {
            "image": torch.randn(
                (
                    cfg.encoder_config.image_encoder_input_size,
                    cfg.encoder_config.image_encoder_input_size,
                    3,
                ),
                dtype=torch.float32,
            ),
        }
        dynamic_axes = None
        output_names = ["image_embeddings"]

    export_onnx(
        cfg=cfg,
        onnx_model=onnx_model,
        dummy_inputs=dummy_inputs,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
        output_file=Path(cfg.output_dir) / "encoder.onnx",
    )


def export_decoder(sam_model: BaseSAM, cfg: DictConfig):
    onnx_model = DecoderOnnxModel(
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        image_encoder_input_size=cfg.encoder_config.image_encoder_input_size,
    )

    embed_dim = onnx_model.prompt_encoder.embed_dim
    embed_size = onnx_model.prompt_encoder.image_embedding_size

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "boxes": torch.rand(4, dtype=torch.float32),
    }
    output_names = ["masks"]

    export_onnx(
        cfg=cfg,
        onnx_model=onnx_model,
        dummy_inputs=dummy_inputs,
        dynamic_axes=None,
        output_names=output_names,
        output_file=Path(cfg.output_dir) / "decoder.onnx",
    )


@task_wrapper
@torch.no_grad()
def export(cfg: DictConfig):
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseSAM = hydra.utils.instantiate(cfg.model).to(cfg.device)
    model.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Exporting encoder to ONNX")
    export_encoder(model, cfg)

    log.info("Exporting decoder to ONNX")
    export_decoder(model, cfg)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="export_onnx.yaml"
)
def main(cfg: DictConfig):
    """Main entry point for exporting.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    export(cfg)


if __name__ == "__main__":
    main()
