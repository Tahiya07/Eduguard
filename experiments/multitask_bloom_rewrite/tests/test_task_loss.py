import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import torch
import torch.nn.functional as F

from task_loss import TASK_TO_ID, task_weighted_causal_lm_loss


class TestTaskWeightedCausalLMLoss(unittest.TestCase):
    def test_matches_manual_sequence_normalized_weighted_loss(self):
        logits = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor(
            [
                [-100, 1, 0, -100],
                [-100, 1, 1, 1],
            ]
        )
        task_ids = torch.tensor([TASK_TO_ID["bloom_rewrite"], TASK_TO_ID["qa"]])
        weights = {TASK_TO_ID["bloom_rewrite"]: 1.5, TASK_TO_ID["qa"]: 1.0}

        actual = task_weighted_causal_lm_loss(logits, labels, task_ids, weights)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shifted_logits.view(-1, 2),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(2, -1)
        mask = shifted_labels.ne(-100)
        per_example = (token_losses * mask).sum(1) / mask.sum(1)
        expected = (per_example * torch.tensor([1.5, 1.0])).mean()
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertIsNotNone(logits.grad)

    def test_rejects_examples_without_supervised_targets(self):
        logits = torch.zeros((1, 2, 3))
        labels = torch.tensor([[-100, -100]])
        task_ids = torch.tensor([TASK_TO_ID["bloom_rewrite"]])
        with self.assertRaisesRegex(ValueError, "supervised target token"):
            task_weighted_causal_lm_loss(
                logits, labels, task_ids, {TASK_TO_ID["bloom_rewrite"]: 1.5}
            )

    def test_rejects_unknown_task_id(self):
        logits = torch.zeros((1, 2, 3))
        labels = torch.tensor([[-100, 1]])
        with self.assertRaisesRegex(ValueError, "No configured loss weight"):
            task_weighted_causal_lm_loss(logits, labels, torch.tensor([99]), {0: 1.5})


if __name__ == "__main__":
    unittest.main()
