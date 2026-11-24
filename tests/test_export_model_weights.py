"""Tests for src/sam_finetuning/export_model_weights.py."""
import torch

from src.sam_finetuning.export_model_weights import model_only_checkpoint


def test_training_checkpoint_keeps_only_the_model_without_the_multi_gpu_prefix():
    weights = torch.ones(2)
    checkpoint = {"model": {"module.mask_decoder.w": weights}, "optimizer": {"state": {}}, "epoch": 7}
    exported = model_only_checkpoint(checkpoint)
    assert list(exported) == ["model"] and list(exported["model"]) == ["mask_decoder.w"]
    assert torch.equal(exported["model"]["mask_decoder.w"], weights)


def test_bare_state_dict_is_wrapped():
    assert list(model_only_checkpoint({"image_encoder.w": torch.zeros(1)})["model"]) == ["image_encoder.w"]
