#!/bin/bash

source script_constants.sh
source script_functions.sh


EXP_NAME="exp1"
# ==============================================================================
# PROGRAM
# ==============================================================================

### Network
# generate $SEED $NODES $TASKS $USERS $COMMUNITIES

### Problem solving
# solve
# arrange

# ### Plotting
# plot_comparison
# plot_convergence

### Helpers
# send_telegram_message

### Looping networks
#for NODES in $(seq 20 20 40); do
#for TASKS in $(seq 20 20 $((NODES*2))); do
#for USERS in $(seq 20 20 $NODES); do
#	:
#done
#done
#done

### Looping algorithms
#for ALGORITHM in ${ALGORITHMS[*]}; do
#	:
#done

### Looping operator versions
#for SAMPLING_VERSION in ${SAMPLING_VERSION_LIST[*]}; do
#for CROSSOVER_VERSION in ${CROSSOVER_VERSION_LIST[*]}; do
#for MUTATION_VERSION in ${MUTATION_VERSION_LIST[*]}; do
#	:
#done
#done
#done

### Looping seeds + thread handling
#for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
#	:
#	pids[${SEED2}]=$!
#done
#
#for pid in ${pids[*]}; do
#	wait $pid
#done

paralel_solve_seed() {
    # Iterations needed for distributing N_EXECUTIONS 
    # among N_PROC (max CPU usage control)
    N_ITER=$((
            N_EXECUTIONS % N_PROC == 0 ?
            N_EXECUTIONS / N_PROC :
            N_EXECUTIONS / N_PROC + 1
        ))
    for ITER in $(seq 1 1 $N_ITER); do
        START=$(((ITER - 1) * N_PROC + 1))
        END=$((
                N_EXECUTIONS <= ITER * N_PROC ?
                N_EXECUTIONS :
                ITER * N_PROC))

        for SEED2 in $(seq $START 1 $END); do
            echo "    $SEED2"
            solve &
            pids[${SEED2}]=$!
        done

        for pid in ${pids[*]}; do
            wait $pid
        done

    done
}

paralel_hybrid_solve_seed() {
    # Run each seed execution sequentially
    for SEED2 in $(seq 13 1 18); do 
        echo "    Running hybrid_solve with seed $SEED2"
        hybrid_solve
    done
}

paralel_table_generation() {
    OBJECTIVES_AUX=(${OBJECTIVES[@]})
    OBJ_PAIRS=()
    for i in $(seq 0 1 $((N_OBJECTIVES-1))); do
        for j in $(seq $((i+1)) 1 $((N_OBJECTIVES-1))); do
            OBJ_PAIRS+=( "${OBJECTIVES_AUX[$i]} ${OBJECTIVES_AUX[$j]}" )
        done
    done
    N_OBJ_PAIRS=${#OBJ_PAIRS[*]}

    N_ITER=$((
            N_OBJ_PAIRS % N_PROC == 0 ?
            N_OBJ_PAIRS / N_PROC :
            N_OBJ_PAIRS / N_PROC + 1
        ))
    for ITER in $(seq 1 1 $N_ITER); do
        START=$(((ITER - 1) * N_PROC))
        END=$((
                N_OBJ_PAIRS <= ITER * N_PROC ?
                N_OBJ_PAIRS - 1 :
                ITER * N_PROC - 1))
        
        for i in $(seq $START 1 $END); do
            IFS=' ' read -r -a OBJECTIVES <<< ${OBJ_PAIRS[$i]}
            N_OBJECTIVES=${#OBJECTIVES[*]}
            echo "OBJECTIVES = (${OBJECTIVES[*]})"
            get_table &
            pids[$i]=$!
        done

        for pid in ${pids[*]}; do
            wait $pid
        done

    done
}

NODES=50
TASKS=50
USERS=25
SEED2=1
N_EXECUTIONS=30

generate
paralel_hybrid_solve_seed
# get_csv_hybrid

# hybrid_solve "_no_dumping"
#get_csv_hybrid "_no_dumping"

#POP_SIZE=300
# arrange_all
#solution_to_ref_points 0.9



