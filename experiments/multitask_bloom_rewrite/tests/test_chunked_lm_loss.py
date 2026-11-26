"""chunked_supervised_loss must equal the model's own causal-LM loss and gradients.

Tiny random Qwen2 + PEFT LoRA on CPU; no downloads, no GPU.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

try:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from chunked_lm_loss import supervised_token_loss
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch/transformers/peft not installed")
class ChunkedLmLossTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        )
        model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False,
        ))
        model.eval()
        return model

    def batch(self):
        torch.manual_seed(1)
        ids = torch.randint(3, 97, (3, 40))
        mask = torch.ones_like(ids)
        labels = ids.clone()
        for row, (prompt_len, length) in enumerate([(25, 40), (2, 30), (30, 36)]):
            labels[row, :prompt_len] = -100
            labels[row, length:] = -100
            mask[row, length:] = 0
            ids[row, length:] = 0
        return ids, mask, labels

    def grads(self, model):
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}

    def check(self, num_items):
        model = self.make_model()
        ids, mask, labels = self.batch()
        model.zero_grad(set_to_none=True)
        ref = model(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=num_items).loss
        ref.backward()
        ref_grads = self.grads(model)
        for chunk in (5, 1024):
            model.zero_grad(set_to_none=True)
            got = supervised_token_loss(model, ids, labels, attention_mask=mask,
                                        num_items_in_batch=num_items, chunk_size=chunk)
            got.backward()
            torch.testing.assert_close(got.detach(), ref.detach(), rtol=1e-5, atol=1e-6)
            for name, grad in ref_grads.items():
                torch.testing.assert_close(self.grads(model)[name], grad, rtol=1e-4, atol=1e-6, msg=name)

    def test_matches_model_loss_with_num_items(self) -> None:
        ids, mask, labels = self.batch()
        self.check(labels[:, 1:].ne(-100).sum() + 7)  # as under gradient accumulation

    def test_matches_model_loss_mean(self) -> None:
        self.check(None)

    def test_no_supervised_tokens_gives_zero(self) -> None:
        model = self.make_model()
        ids, mask, labels = self.batch()
        loss = supervised_token_loss(model, ids, torch.full_like(labels, -100), attention_mask=mask)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
