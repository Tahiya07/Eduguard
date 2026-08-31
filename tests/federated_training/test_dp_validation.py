"""Tests for DP validation gates (synthetic clipping/accounting; no full model download)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


def test_clipping_norm_cases():
  """Synthetic post-clipping norms obey norm <= C."""
  C = 1.0
  cases = {
      "below": torch.tensor([0.3, 0.4]),
      "equal": torch.tensor([0.6, 0.8]),
      "above": torch.tensor([3.0, 4.0]),
  }
  for label, grad in cases.items():
      norm = grad.norm(2).item()
      scale = min(1.0, C / (norm + 1e-12))
      clipped = grad * scale
      clipped_norm = clipped.norm(2).item()
      assert clipped_norm <= C + 1e-6, label
      if norm > 0:
          expected_scale = C / norm if norm > C else 1.0
          assert math.isclose(scale, expected_scale, rel_tol=1e-5)


def test_accounting_epsilon_monotonicity():
  pytest.importorskip("opacus")
  from opacus.accountants import RDPAccountant

  accountant = RDPAccountant()
  noise = 1.1
  sample_rate = 0.25
  delta = 1e-5
  accountant.history = [(noise, sample_rate, 50)]
  eps_50 = accountant.get_epsilon(delta)
  accountant.history.append((noise, sample_rate, 100))
  eps_100 = accountant.get_epsilon(delta)
  assert eps_100 > eps_50

  accountant2 = RDPAccountant()
  accountant2.history = [(2.0, sample_rate, 100)]
  eps_high_noise = accountant2.get_epsilon(delta)
  accountant3 = RDPAccountant()
  accountant3.history = [(0.5, sample_rate, 100)]
  eps_low_noise = accountant3.get_epsilon(delta)
  assert eps_low_noise > eps_high_noise


def test_dp_lock_absent_blocks_federated_dp():
  from training.federated import dp as fed_dp
  from training.paths import ARTIFACTS_PRIVACY

  lock = ARTIFACTS_PRIVACY / "dp_bloom_validated_v1.json"
  if lock.is_file():
      data = __import__("json").loads(lock.read_text(encoding="utf-8"))
      if data.get("validation_gate_passed"):
          pytest.skip("DP lock already passed in this workspace")
  with pytest.raises(SystemExit):
      fed_dp.load_dp_lock()
