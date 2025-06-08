#!/bin/bash

# ==============================================================================
# CONSTANTS
# ==============================================================================
SEED=722

NODES=50
TASKS=50
USERS=25

COMMUNITIES=true

N_REPLICAS=$NODES
MUTATION_PROB_MOVE=0.2
MUTATION_PROB_MOVE_LIST=(0.1 0.2)
MUTATION_PROB_CHANGE=0.1
MUTATION_PROB_CHANGE_LIST=(0.1 0.2)
MUTATION_PROB_BINOMIAL=0.1
MUTATION_PROB_BINOMIAL_LIST=(0.025 0.05)

POP_SIZE=100 ##for hibrids
#POP_SIZE = 400 ##for simple execution


#POP_SIZES=($(seq 200 50 300))
# POP_SIZES=($(seq 200 50 100))
# N_GEN=400
N_GEN=100


ALGORITHM='NSGA2'
ALGORITHMS=('NSGA2') # 'NSGA3' 'UNSGA3' 'SMSEMOA') # 'RVEA')
# Available: 'NSGA2' 'NSGA3' 'UNSGA3' 'CTAEA' 'SMSEMOA' 'RVEA' 'RNSGA2' 'RNSGA3'
# Not implemented: 'AGEMOEA'
N_ALGORITHMS=${#ALGORITHMS[@]}

SAMPLING_VERSION=0
SAMPLING_VERSION_LIST=(0)
CROSSOVER_VERSION=2
CROSSOVER_VERSION_LIST=(2)
MUTATION_VERSION=1
MUTATION_VERSION_LIST=(1)

N_PARTITIONS=0

REF_POINTS_ALGORITHM='ALL'
LAMBDA_LIST=($(LANG=en_US seq 0.1 0.2 1))

# 'ntw_utilization' 'occupation' 'nodes' 'hops')
OBJECTIVES=('distance' 'occ_variance' 'pw_consumption') 
N_OBJECTIVES=${#OBJECTIVES[@]}


N_PROC=5
SEED2=A
SEED2=1

# HYBRID_POP_SIZE=100
# HYBRID_N_GEN=600

HYBRID_POP_SIZE=100
HYBRID_N_GEN=100


HYBRID_GEN_STEPS=(25 50 75)
GEN_STEP=3
# HYBRID_GEN_STEPS=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600)
# HYBRID_ALGORITHMS=('NSGA2' 'NSGA3' 'UNSGA3' 'SMSEMOA')
HYBRID_ALGORITHMS=('NSGA2'  'SMSEMOA')
# HYBRID_ALGORITHMS=('NSGA2' 'CTAEA' )
# HYBRID_ALGORITHMS=('NSGA2' 'NSGA3')
HYBRID_SOL_DUMPING_ALL="$(python3 -c "print(
        [
            [
                [
                    1 for _ in range(${#HYBRID_ALGORITHMS[@]})
                ] for _ in range(${#HYBRID_ALGORITHMS[@]})
            ] for _ in range(${#HYBRID_GEN_STEPS[@]})
        ]
    )")"
echo $HYBRID_SOL_DUMPING_ALL
# MYHYBRID_SOL_DUMPING="[[[1,1],[1,1]],[[1,0],[1,1]],[[1,1],[1,1]]]"
# echo $MYHYBRID_SOL_DUMPING

HYBRID_SOL_DUMPING_NONE="$(python3 -c "print(
        [
            [
                [
                    0 for _ in range(${#HYBRID_ALGORITHMS[@]})
                ] for _ in range(${#HYBRID_ALGORITHMS[@]})
            ] for _ in range(${#HYBRID_GEN_STEPS[@]})
        ]
    )")"
HYBRID_SOL_DUMPING="$HYBRID_SOL_DUMPING_ALL"
# HYBRID_SOL_DUMPING="$MYHYBRID_SOL_DUMPING"

HYBRID_TABLE_GEN_STEPS=(1 $(seq 5 5 $((HYBRID_N_GEN - 1))) $HYBRID_N_GEN)

# PREFIXES
NTW_PREFIX="data/networks"
SOL_PREFIX="data/solutions" 
ALY_PREFIX="data/analysis"

PREFIX="data/solutions/P$POP_SIZE-G$N_GEN/MM$MUTATION_PROB_MOVE-MC$MUTATION_PROB_CHANGE/new_crossover"
PREFIX2="data/solutions/P$POP_SIZE-G$N_GEN/MM$MUTATION_PROB_MOVE-MC$MUTATION_PROB_CHANGE/communities"


