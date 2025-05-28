#!/bin/bash

source script_constants.sh
source script_functions.sh

# ==============================================================================
# PROGRAM
# ==============================================================================

### Network
# generate $SEED $NODES $TASKS $USERS $COMMUNITIES

### Problem solving
# solve
# arrange

### Plotting
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

### Looping seeds + thread handling
#for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
#	:
#	pids[${SEED2}]=$!
#done
#
#for pid in ${pids[*]}; do
#	wait $pid
#done

NODES=50
TASKS=50
USERS=25

SEED2=A
#POP_SIZES=($(seq 50 50 300))
#for ALGORITHM in ${ALGORITHMS[*]}; do
#for POP_SIZE in ${POP_SIZES[*]}; do
#	echo "POP_SIZE = $POP_SIZE"
#	plot_comparison # population "with algorithm $ALGORITHM"
#done

SEED2=1
#POP_SIZE=200
#N_GEN=400
#plot_comparison

N_EXECUTIONS=1

     # -i "data/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3-UNSGA3-SMSEMOA/$ALG""_$S2""_100-200_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt" \
    #  --hybrid_legend NSGA2 NSGA3 UNSGA3 SMSEMOA &
for ALG in ${HYBRID_ALGORITHMS[*]}; do
    for S2 in $(seq 1 1 $N_EXECUTIONS); do
        uv run hybrid_plot.py plot \
            -i "data/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3/$ALG""_$S2""_100-200_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt" \
            --objectives distance occ_variance pw_consumption \
            --n_objectives 3 \
            --stack \
            --title "$ALG ecosystem with seed $S2" \
            --hybrid_legend NSGA2 NSGA3 &
        pids[${S2}]=$!
    done

    for pid in ${pids[*]}; do
        wait $pid
    done

    #for S2 in $(seq 1 1 $N_EXECUTIONS); do
    #    python3 hybrid_plot.py plot \
    #        -i "data/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3-UNSGA3-SMSEMOA/$ALG""_$S2""_050-050_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt" \
    #        --objectives distance occ_variance pw_consumption \
    #        --n_objectives 3 \
    #        --title "$ALG ecosystem with seed $S2" \
    #        --hybrid_legend NSGA2 NSGA3 UNSGA3 SMSEMOA &
    #    pids[${S2}]=$!
    #done

    #for pid in ${pids[*]}; do
    #    wait $pid
    #done

done


