# Statistical Analysis for Genetic Algorithm Comparison

This directory contains scripts to perform statistical analysis on the performance metrics of competing genetic algorithms using Wilcoxon rank sum tests with a significance level of 0.05.

## Overview

The analysis compares the following metrics across all algorithms:
- **GD** (Generational Distance) - Lower is better
- **IGD** (Inverted Generational Distance) - Lower is better  
- **HV** (Hypervolume) - Higher is better
- **S** (Spread) - Lower is better
- **STE** (Spacing to Extent) - Lower is better

## Files

### Main Scripts
- `run_statistical_analysis.py` - Main runner script with options
- `statistical_analysis_detailed.py` - Comprehensive analysis with multiple testing correction
- `wilcoxon_statistical_analysis.py` - Basic analysis without multiple testing correction

### Configuration
- `script_constants.sh` - Contains algorithm configurations and parameters
- `utils.py` - Utility functions for data loading

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
```bash
# Run comprehensive analysis (recommended)
python run_statistical_analysis.py --detailed

# Or run basic analysis
python run_statistical_analysis.py --simple

# Or just run (defaults to detailed)
python run_statistical_analysis.py
```

## Analysis Types

### Comprehensive Analysis (`--detailed`)
- **Wilcoxon rank sum tests** for all algorithm pairs
- **Multiple testing correction** using Benjamini-Hochberg method
- **Friedman tests** to check for overall differences
- **Effect size calculations** (rank-biserial correlation)
- **Comprehensive reporting** with detailed statistics
- **Visualizations** (heatmaps and boxplots)

### Basic Analysis (`--simple`)
- **Wilcoxon rank sum tests** for all algorithm pairs
- **Basic reporting** without multiple testing correction
- **Simple visualizations**

## Output Files

### Comprehensive Analysis Outputs
- `statistical_analysis_detailed_results.csv` - All test results with corrected p-values
- `statistical_analysis_significant_results.csv` - Only significant results
- `statistical_analysis_summary.csv` - Summary statistics
- `statistical_analysis_friedman_results.csv` - Friedman test results
- `wilcoxon_heatmap.png` - Heatmap of p-values
- `metrics_boxplots.png` - Boxplots of metric distributions

### Basic Analysis Outputs
- `wilcoxon_test_results.csv` - All test results
- `wilcoxon_heatmap.png` - Heatmap of p-values
- `metrics_boxplots.png` - Boxplots of metric distributions

## Understanding the Results

### P-Values
- **p < 0.001**: Highly significant (***)
- **p < 0.01**: Very significant (**)
- **p < 0.05**: Significant (*)

### Effect Size (Rank-Biserial Correlation)
- **|r| > 0.5**: Large effect
- **|r| > 0.3**: Medium effect
- **|r| > 0.1**: Small effect

### Multiple Testing Correction
When performing many statistical tests, the chance of false positives increases. The Benjamini-Hochberg correction controls the false discovery rate.

## Data Requirements

The analysis expects the following data files:
- `results/table_standard_{POP_SIZE}_{N_GEN}.csv` - Standard algorithm results
- `results/table_hybrids_{HYBRID_POP_SIZE}_{HYBRID_N_GEN}.csv` - Hybrid algorithm results

These files should contain columns:
- `Algorithm`: Algorithm name
- `Seed`: Replica/seed number
- `Generation`: Generation number
- `GD`, `IGD`, `HV`, `S`, `STE`: Metric values

## Example Usage

```python
# Run from command line
python run_statistical_analysis.py --detailed

# Or import and use programmatically
from statistical_analysis_detailed import main
results, friedman_results = main()
```

## Interpretation Guide

### Significant Results
When you find significant differences:
1. **Check the effect size** - Large effects are more meaningful
2. **Consider the practical significance** - Small differences may not matter
3. **Look at the direction** - Which algorithm performs better?

### Non-Significant Results
When no significant differences are found:
1. **Check sample sizes** - Small samples may lack power
2. **Consider effect sizes** - Even non-significant effects may be meaningful
3. **Look at confidence intervals** - They show the range of possible effects

## Troubleshooting

### Common Issues

1. **"No metrics data files found"**
   - Ensure the CSV files exist in the `results/` directory
   - Check that the file names match the expected pattern

2. **"Insufficient data"**
   - Ensure you have at least 3 replicates per algorithm
   - Check for missing or NaN values in the data

3. **"Error in Wilcoxon test"**
   - Usually indicates identical data between algorithms
   - Check if the algorithms actually produced different results

### Data Quality Checks
```python
# Check data availability
import pandas as pd
df = pd.read_csv('results/table_standard_400_600.csv')
print(df['Algorithm'].value_counts())
print(df.groupby('Algorithm')['Seed'].nunique())
```

## Advanced Usage

### Custom Analysis
You can modify the scripts to:
- Change the significance level (default: 0.05)
- Add different statistical tests
- Customize visualizations
- Filter specific algorithms or metrics

### Batch Processing
For multiple experiments:
```bash
# Run analysis for different configurations
for config in config1 config2 config3; do
    python run_statistical_analysis.py --detailed
    mv statistical_analysis_* results_${config}/
done
```

## References

- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701. 