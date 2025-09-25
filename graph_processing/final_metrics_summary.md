# Vision Pipeline Final Metrics Report

## Pipeline Execution Summary
- **Model Used**: GLM-4.1V-9B-Thinking
- **Total Files Processed**: 933
- **Datasets Processed**: All 5 datasets (synergy, cytox, magnetic, nanozymes, seltox)
- **Processing Time**: Completed successfully with exit code 0

## Extraction Performance Metrics

### Overall Statistics
- **Average Precision**: 0.250 (25%)
- **Average Recall**: 0.083 (8.3%)
- **Average F1-Score**: 0.125 (12.5%)
- **Average Accuracy**: 0.083 (8.3%)

### Dataset-Specific Results

#### Synergy Dataset
- **Train Set**: 69 files processed from 9 ground truth files
  - NP_size_avg_nm: No successful extractions (0/8)
  - zeta_potential_mV: 1 successful extraction (1/3, precision=100%, recall=33%)
  - CI: No ground truth data available

- **Test Set**: 18 files processed from 3 ground truth files
  - All parameters: No successful extractions

#### Cytox Dataset
- **Train/Test Sets**: Files processed but no ground truth matches found
  - IC50_ug_per_ml: No data
  - Cell_type: No data
  - Cell_viability: No data

#### Magnetic Dataset
- **Train Set**: 602 files processed from 36 ground truth files
- **Test Set**: 152 files processed from 24 ground truth files
  - Ms_emu_per_g, Mr_emu_per_g, Hc_Oe: No ground truth data available

#### Nanozymes Dataset
- **Train Set**: 632 files processed from 181 ground truth files
- **Test Set**: 158 files processed from 44 ground truth files
  - Km_value, Vmax_value, Kcat_value: No ground truth data available

#### Seltox Dataset
- **Train/Test Sets**: Files processed but no ground truth matches found
  - IC50_ug_per_ml, Cell_type, SI: No data

## Key Findings

1. **Low Extraction Rate**: The VLM extracted very few quantitative parameters successfully
2. **Best Performance**: Only synergy dataset showed some success (1 zeta potential value extracted correctly)
3. **Missing Ground Truth**: Many datasets have no matching ground truth values in the CSV files
4. **Processing Errors**: 754 __MACOSX system files encountered (non-critical)

## Next Steps

1. **Gemma-3-27b Testing**: Still pending - need to run the same pipeline with Gemma model for comparison
2. **Prompt Optimization**: Current prompts may need refinement for better extraction
3. **Ground Truth Verification**: Need to verify CSV files contain correct mappings to PDF files
4. **Parameter Extraction Enhancement**: Consider more targeted prompts for specific parameter types