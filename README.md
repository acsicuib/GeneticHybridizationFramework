# Genetic Hybridization Placement
GeneticHybridizationPlacement is a Python-based ecosystem for the optimization of resource placement in fog computing environments using Genetic Algorithms (GAs). The framework supports both single-algorithm and hybrid-algorithm evolutionary strategies, enabling comprehensive experimentation and benchmarking.

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
### Dependencies
We recommend the use of uv python package manager. Install all required libraries with:
```
uv sync
source .venv/bin/activate
```
## Experiment running
Define experiment parameter via shell scripts.
```bash
.\run_single.sh
normalize_solutions.py
```

or run Hybrids configurations
```bash
.\run_hybrids.sh
```

Compute reference points, compare single and hybrids results and plot
Situacion actual en cloudlab /results: 
```bash
uv run compute_unified_PF.py --control single 
uv run compute_unified_PF.py --control hybrids 
uv run compute_unified_PF.py --control merge ### Ejecutando este punto en el servidor con las 30 replicas
uv run compute_metrics_hybridization.py
uv run compute_metrics_standard.py
```
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
