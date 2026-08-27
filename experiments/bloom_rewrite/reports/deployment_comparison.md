# Deployment comparison

**Status: pending.** No merged HF checkpoint and no experimental GGUF has been produced.

Quantization must wait until the HF/merged generator is evaluated on the held-out test set. Then convert to GGUF Q4_K_M and re-evaluate the **same** test items.

Measure for **both** candidate GGUFs during actual CPU inference (not from file size alone):

- GGUF file size
- startup time
- first-token latency
- generation latency
- peak RSS
- USS if available
- CPU threads
- context size

Deployment target: CPU-only, offline, approximately ≤1 GB **measured memory**, not “GGUF file is under 1 GB”.

Do not replace the production GGUF from this experiment until a model is selected on measured quality **and** measured cost.
