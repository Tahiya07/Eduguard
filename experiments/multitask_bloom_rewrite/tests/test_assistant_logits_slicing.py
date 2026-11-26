"""assistant_logits_only must give the same loss and LoRA gradients as full logits.

Tiny random Qwen2 + PEFT LoRA on CPU; no downloads, no GPU.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from loss_masking import slice_to_assistant_logits  # noqa: E402

try:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch/transformers/peft not installed")
class AssistantLogitsSlicingTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        )
        model = Qwen2ForCausalLM(cfg)
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False,
        ))
        model.eval()
        return model

    def batch(self):
        # Right-padded batch; prompts of different lengths, answers at the end.
        torch.manual_seed(1)
        ids = torch.randint(3, 97, (3, 40))
        mask = torch.ones_like(ids)
        labels = ids.clone()
        for row, (prompt_len, length) in enumerate([(25, 40), (12, 30), (30, 36)]):
            labels[row, :prompt_len] = -100
            labels[row, length:] = -100
            mask[row, length:] = 0
            ids[row, length:] = 0
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    def loss_and_grads(self, model, inputs, num_items):
        model.zero_grad(set_to_none=True)
        loss = model(**inputs, num_items_in_batch=num_items).loss
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        return loss.detach(), grads

    def test_slice_keeps_every_supervised_label(self) -> None:
        inputs = self.batch()
        sliced = slice_to_assistant_logits(inputs)
        self.assertEqual(sliced["logits_to_keep"], 40 - 11)
        self.assertEqual(int(sliced["labels"].ne(-100).sum()), int(inputs["labels"].ne(-100).sum()))

    def test_identical_loss_and_gradients(self) -> None:
        model = self.make_model()
        inputs = self.batch()
        num_items = inputs["labels"][:, 1:].ne(-100).sum()
        full_loss, full_grads = self.loss_and_grads(model, inputs, num_items)
        cut_loss, cut_grads = self.loss_and_grads(model, slice_to_assistant_logits(inputs), num_items)
        torch.testing.assert_close(cut_loss, full_loss, rtol=1e-5, atol=1e-6)
        for name, grad in full_grads.items():
            torch.testing.assert_close(cut_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)

    def test_noop_without_prompt_mask_or_labels(self) -> None:
        inputs = self.batch()
        inputs["labels"][0, :] = inputs["input_ids"][0, :]
        self.assertIs(slice_to_assistant_logits(inputs), inputs)
        no_labels = {"input_ids": inputs["input_ids"]}
        self.assertIs(slice_to_assistant_logits(no_labels), no_labels)


if __name__ == "__main__":
    unittest.main()
