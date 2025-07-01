# Rebuttal Materials

This folder contains supporting evidence and scripts used to evaluate and compare the **zero-shot** and **few-shot** performance of the extraction pipeline described in our submission. The evaluations are based on 7 scientific articles from the nanomaterials domain.

## Folder Structure

- **`calc/`**  
  Contains the evaluation script used to compute precision, recall, and F1 scores across all target fields.

- **`results/`**  
  Includes:
  - `zero_shot_results.txt` and `few_shot_results.txt`: Output metrics (TP, FP, FN, precision, recall, F1) for each target field.
  - Real extraction outputs from both few-shot and zero-shot modes.

- **`test/`**  
  Full set of **7 evaluation articles**, including main texts and associated supplementary information (SI), used for both zero-shot and few-shot testing.

- **`example/`**  
  A randomly selected scientific article used as an **in-context prompt** for few-shot experiments.
