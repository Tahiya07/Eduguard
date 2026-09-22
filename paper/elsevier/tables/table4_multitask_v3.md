# Final multitask v3 test performance (frozen test). Classifier is not human ground truth.

| Task | N | Metric A | Metric B | Metric C | Notes |
| --- | --- | --- | --- | --- | --- |
| Bloom rewrite | 1536 | 0.8880 | 0.8768 | 0.7070 | merged 0.5B classifier |
| Bloom rewrite (integrated) | 1536 | 0.9518 | 0.9443 | 0.7624 | FedProx r20 classifier |
| QA | 5285 | 0.5784 (EM) | 0.7854 (F1) | — | same generations |
| Summarization | 1500 | 0.2491 (R1) | 0.0573 (R2) | 0.1496 (RL) | same generations |
