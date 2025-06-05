#!/bin/bash

source script_constants.sh
source script_functions.sh

###############################################################################
# Constants
###############################################################################
EXP_NAME="exp_single"

# OBJECTIVES=(
#     'distance'
#     'occ_variance'
#     'pw_consumption'
#     'ntw_utilization'
#     'occupation'
#     'nodes'
#     'hops'
# )
OBJECTIVES=('distance' 'occ_variance' 'pw_consumption')

N_OBJECTIVES=${#OBJECTIVES[@]}

OBJ_UPPERB_LIST=( 3  )
POP_SIZES=(       400)
N_GENS=(          600)

GEN_STEP=5 # for csv, print each 5 generations

# Executions and max paralel processes
N_EXECUTIONS=10
N_PROC=5



################################################################################
# FUNCTIONS
################################################################################
get_files_exp1() {
    NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
    SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM \
            ${OBJECTIVES[@]})"

    ALG_FILES=()
    i=0
    for ALG in ${ALGORITHMS[*]}; do
        for SD in $(seq 1 1 $N_EXECUTIONS); do
            SOL_NAME="$(get_solution_filename $ALG $SD $POP_SIZE $N_GEN \
                    $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION \
                    $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE \
                    $MUTATION_PROB_BINOMIAL)"
            ALG_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME" 
            i=$((i+1))
        done
    done
    echo ${ALG_FILES[*]}
}

get_csv_exp1() {
    FILES=$(get_files_exp1)
    NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
    SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM \
            ${OBJECTIVES[@]})"

    local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS \
        ${OBJECTIVES[@]})"
    local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
    local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
    if [ -f "$REF_FILE" ]; then
        local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
        local REF_POINTS_OPT=--ref_points
    else
        local REF_POINTS_STRING=
        local REF_POINTS_OPT=
    fi

    SUFFIX=""$ALGORITHM"_$POP_SIZE:$N_GEN"

    mkdir -p "$ALY_PREFIX/$SOL_PATH/$EXP_NAME"
    python3 main.py analyze \
        --objectives ${OBJECTIVES[*]} $REF_POINTS_OPT $REF_POINTS_STRING \
        --network "$NTW_PREFIX/$NTW_NAME" \
        -i $FILES \
        --alg_names ${ALGORITHMS[*]} \
        --seed_values $(seq 1 1 $N_EXECUTIONS) \
        --pop_values $POP_SIZE \
        --gen_values $(seq $GEN_STEP $GEN_STEP $N_GEN) \
        --output "$ALY_PREFIX/$SOL_PATH/$EXP_NAME/table_$SUFFIX.csv"
}

get_all_csv() {
    # Create a copy of the list of objectives
    OBJECTIVES_AUX=(${OBJECTIVES[@]})

    # From 2 to $N_OBJECTIVES number of objectives
    LENGTH=${#OBJ_UPPERB_LIST[@]}
    for IDX in $(seq 0 1 $((LENGTH - 1))); do

        # Select subset of objectives with upper bound
        OBJ_UPPERB=${OBJ_UPPERB_LIST[$IDX]}
        OBJECTIVES=(${OBJECTIVES_AUX[@]::$OBJ_UPPERB})
        echo "OBJECTIVES = (${OBJECTIVES[@]})"

        # Load generations and pop size for current number of objectives
        N_GEN=${N_GENS[$IDX]}
        POP_SIZE=${POP_SIZES[$IDX]}
        echo "  $POP_SIZE:$N_GEN"

        get_csv_exp1

    done
}

play_with_objectives() {
    # Create a copy of the list of objectives
    OBJECTIVES_AUX=(${OBJECTIVES[@]})

    # From 2 to $N_OBJECTIVES number of objectives
    LENGTH=${#OBJ_UPPERB_LIST[@]}
    for IDX in $(seq 0 1 $((LENGTH - 1))); do

        # Select subset of objectives with upper bound
        OBJ_UPPERB=${OBJ_UPPERB_LIST[$IDX]}
        OBJECTIVES=(${OBJECTIVES_AUX[@]::$OBJ_UPPERB})
        echo "OBJECTIVES = (${OBJECTIVES[@]})"

        # Load generations and pop size for current number of objectives
        N_GEN=${N_GENS[$IDX]}
        POP_SIZE=${POP_SIZES[$IDX]}
        echo "  $POP_SIZE:$N_GEN"

        for ALGORITHM in ${ALGORITHMS[*]}; do
            echo "    ALGORITHM = $ALGORITHM"

            # Load number of iterations having:
            #   - N_EXECUTIONS total executions
            #   - N_PROC max paralel processes
            N_ITER=$((
                    N_EXECUTIONS % N_PROC == 0 ?
                    N_EXECUTIONS / N_PROC :
                    N_EXECUTIONS / N_PROC + 1
                ))
            for ITER in $(seq 1 1 $N_ITER); do

                # Calculate start iteration index and end iteration index
                START=$(((ITER - 1) * N_PROC + 1))
                END=$((
                        N_EXECUTIONS <= ITER * N_PROC ?
                        N_EXECUTIONS :
                        ITER * N_PROC
                    ))
                echo "      $START-$END"

                for SEED2 in $(seq $START 1 $END); do
                    echo "        $SEED2"
                    solve &
                    pids[$i]=$!
                    i=$((i+1))
                done

                for pid in ${pids[*]}; do
                    wait $pid
                done
            done

        done
    done

}



###############################################################################
# INFRASTRUCTURE SIZE 1
###############################################################################
NODES=50
TASKS=50
USERS=25

generate # generate network
play_with_objectives
get_all_csv



###############################################################################
# INFRASTRUCTURE SIZE 2
###############################################################################
# NODES=100
# TASKS=50
# USERS=25

# generate # generate network
# play_with_objectives
# get_all_csv



###############################################################################
# INFRASTRUCTURE SIZE 3
###############################################################################
# NODES=50
# TASKS=100
# USERS=25

# generate # generate network
# play_with_objectives
# get_all_csv



###############################################################################
# INFRASTRUCTURE SIZE 4
###############################################################################
# NODES=50
# TASKS=50
# USERS=50

# generate # generate network
# play_with_objectives
# get_all_csv


