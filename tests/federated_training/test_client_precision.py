"""Training precision flags must not combine full-fp16 weights with fp16 AMP."""

from __future__ import annotations

from unittest.mock import patch

import torch

from training.federated.client import _resolve_training_precision


def test_cpu_precision_is_fp32_without_amp():
    with patch("training.federated.client.torch.cuda.is_available", return_value=False):
        dtype, fp16, bf16 = _resolve_training_precision()
    assert dtype == torch.float32
    assert fp16 is False
    assert bf16 is False


def test_cuda_fp16_amp_uses_fp32_master_weights():
    with patch("training.federated.client.torch.cuda.is_available", return_value=True):
        with patch("training.federated.client.torch.cuda.is_bf16_supported", return_value=False):
            dtype, fp16, bf16 = _resolve_training_precision()
    assert dtype == torch.float32
    assert fp16 is True
    assert bf16 is False


def test_cuda_bf16_uses_bf16_weights_without_fp16_amp():
    with patch("training.federated.client.torch.cuda.is_available", return_value=True):
        with patch("training.federated.client.torch.cuda.is_bf16_supported", return_value=True):
            dtype, fp16, bf16 = _resolve_training_precision()
    assert dtype == torch.bfloat16
    assert fp16 is False
    assert bf16 is True
