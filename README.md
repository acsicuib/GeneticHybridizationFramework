# Genetic Hybridization Placement
GeneticHybridizationPlacement is a Python-based ecosystem for the optimization of resource placement in fog computing environments using Genetic Algorithms (GAs). The framework supports both standalone (aka single-algorithm) and hybrid-algorithm evolutionary strategies, enabling comprehensive experimentation and benchmarking.

## Project Overview
This project provides a flexible environment to:
- Run single GA algorithms (e.g., NSGA2, NSGA3, UNSGA3, SMSEMOA) independently on resource placement problems.
- Run hybrid GAs, where multiple algorithms cooperate or compete within the same evolutionary process, allowing for hybridization strategies and advanced metaheuristics.
- Analyze and compare the performance of single and hybrid approaches using a unified set of metrics and statistical tools.
- Automate experiments across different network topologies, objective combinations, and random seeds, with support for parallel execution and reproducibility.

## Key Features
- Single Mode: Execute and analyze standard GAs individually for baseline performance.
- Hybrid Mode: Combine several GAs in a single run, leveraging their strengths for improved optimization.
- Experiment Automation: Shell scripts and Python tools for batch runs, normalization, and result aggregation.
- Statistical Analysis: Built-in scripts for statistical comparison, Pareto front computation, and visualization.

## Getting Started
Clone the project
```
git clone ..
```
### Dependencies
We recommend the use of uv python package manager. Install all required libraries with:
```
uv sync
```
## Experiment running
The execution last long time the scripts and ppost-analysis is seperated in multiples files. It's in your on the orchestation and distribution in differents computational devices.

The parametrization realay in two files:
- mainly, in script_constants.sh
- some residual parameters in file: resopt/param/parameters.py

Modify these two bash scripts according with your experimentation:
- run standalone or single GA algorithms:
```bash
.\run_single.sh
```

- or run Hybrid configuration:
```bash
.\run_hybrids.sh
```

Once you have the results, you can compute the final reference points, which will be used to calculate performance metrics across different configurations and provide a baseline comparison between single and hybrid approaches. There are additional scripts available for generating various plots.
  
```bash
# Extract the reference points from:
uv run compute_unified_PF.py --control single 
uv run compute_unified_PF.py --control hybrids 
# Merge reference points from both execution, internally check if the previous files are computed
uv run compute_unified_PF.py --control merge 
# Merge hybridization population among scenarios
uv run merge_hybridization_cases.py 
# Compute the metrics in hybrid case and standalone scenario (aka standard term)
uv run compute_metrics_hybridization.py
uv run compute_metrics_standard.py 
# Plots
## a simple GD, IGD... convergence plots
uv run do_algorithms_plot_with_replicates.py
# Performance and plot the Wilcoxon test
uv run wilcoxon_statistical_analysis.py
# Plot the exchange of genetical material: stacked and line plot
uv run do_plot_genetical_crossing.py
uv run do_plot_genetical_crossing_line.py
```

**Important:**  
- Follow the sequence, as it is necessary to normalize the results.
- You can also distribute the execution among different devices and only transfer the output data to maintain the performance of other scripts.
- These scripts use `script_constants.sh` to obtain reference variables (population size, number of generations, etc.), but you can include your own variables internally for testing.


## Directory Structure
- resopt/ – Core optimization, problem definitions, and plotting modules.
- data/ – Output directories for experiment results.
- scripts/ – Shell scripts for running and managing experiments.
- test/ – Unit and integration tests.

## Citation
If you use this ecosystem in your research, please cite the corresponding paper or contact the authors for more information.

```text
pending
```

## Project
HIDDEMS – PID2021-128071OB-I00 MICIU/AEI/10.13039/501100011033 and FEDER,UE
(HIDDEMs)[https://hiddems.uib.cat]

## Note
This project and repository are under continuous development.
# Acknowledge 
This project extends the [original report](https://github.com/sergivivo/ResourcesOptimization_01)  in terms of experimentation analysis and plotting. 