# Final multitask reproducibility

Use Python 3.10+ with the packages in requirements-train.txt.

The final generator pipeline is:

1. BloomShift bloomshift_final_candidate
2. SQuAD 1.1
3. FiscalNote BillSum
4. 40/30/30 task sampling for training
5. 8192-token SFT context limit
6. strict dataset QC
7. 1.5B training sanity check
8. resource gate
9. final 1.5B LoRA training

Run run_training.ps1 for the complete sequence.

The script intentionally does not invoke the 0.5B Bloom classifier dataset.
