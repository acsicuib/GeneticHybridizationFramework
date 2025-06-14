# ResourcesOptimization_01
Application developed in Python for genetic optimisation of resources in the context of fog computing.

# Project Mention


## Dependencies

Install libraries with:
```
uv sync
source .venv/bin/activate
```
## Experiment running

1º Single executions
```
.\run_single.sh
normalize_solutions.py
```

2º Hybrids
.\run_hybrids.sh

3º Compute the final pf from all single and hybrids
uv run compute_unified_PF.py

un run compute_indicator.py





## REMOVE
tar czvf data.tar.gz data/*
scp isaac@deepblue:/hdd/isaac/projects/GeneticHybridizationPlacement/data.tar.gz .
tar xzvf data.tar.gz