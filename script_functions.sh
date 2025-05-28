#!/bin/bash

# ==============================================================================
# FUNCTIONS
# ==============================================================================
zeropad_left() {
	printf "%0$2d\n" $1
}

get_network_filename() {
	local SEED=$(zeropad_left $1 3)
	local NODES=$(zeropad_left $2 3)
	local TASKS=$(zeropad_left $3 3)
	local USERS=$(zeropad_left $4 3)
	local COMMUNITIES=$5

	if [ $COMMUNITIES = true ]; then
		local SUFFIX="C"
	else
		local SUFFIX="H"
	fi

	echo "ntw_"$SEED"_"$NODES"-"$TASKS"-"$USERS"_"$SUFFIX
}

generate() {
	mkdir -p "$NTW_PREFIX"

	if [ $COMMUNITIES = true ]; then
		local C_OPT=--communities
	else
		local C_OPT=
	fi

	local NTW_FILENAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"

	python3 main.py --seed $SEED generate \
		--n_nodes $NODES --n_tasks $TASKS --n_users $USERS \
		$C_OPT -o "$NTW_PREFIX/$NTW_FILENAME"
}

get_solution_path() {
	local NTW_NAME=$1
	local N_REPLICAS=$(zeropad_left $2 3)
	local ALGORITHM=$3
	shift; shift; shift
	local OBJECTIVES=("$@")

	if [ $ALGORITHM = "ILP" ]; then
		local ALG_TYPE="ILP"
	else
		local ALG_TYPE="Genetics"
	fi
	
	local prev_ifs=$IFS
	IFS='-'; local OBJ_STRING="${OBJECTIVES[*]}"; IFS=$prev_ifs

	echo "$NTW_NAME/obj_$OBJ_STRING/Replicas$N_REPLICAS/$ALG_TYPE"
}

get_solution_filename() {
	local ALGORITHM=$1
	if [ $ALGORITHM = "ILP" ]; then
		echo "ILP.txt"
	else
		local SEED=$2
		local POP_SIZE=$(zeropad_left $3 3)
		local N_GEN=$(zeropad_left $4 3)
		local SV=$5
		local CV=$6
		local MV=$7
		local MM=$8
		local MC=$9
		local MB=${10}
        local SUFFIX=${11}
        local SUFFIX="${SUFFIX:=}"
        
		echo $ALGORITHM"_"$SEED"_"$POP_SIZE"-"$N_GEN"_SV"$SV"-CV"$CV"-MV"$MV"_MM"$MM"-MC"$MC"-MB"$MB$SUFFIX".txt"
	fi
}

get_ref_points_path() {
	local NTW_NAME=$1
	local N_REPLICAS=$(zeropad_left $2 3)
	shift; shift
	local OBJECTIVES=("$@")

	local prev_ifs=$IFS
	IFS='-'; local OBJ_STRING="${OBJECTIVES[*]}"; IFS=$prev_ifs

	echo "$NTW_NAME/obj_$OBJ_STRING/Replicas$N_REPLICAS/RefPoints"
}

get_ref_points_filename() {
	local ALGORITHM=$1
	echo "rp_$ALGORITHM.txt"
}

solve() {
	local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	local SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
	local SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	mkdir -p "$SOL_PREFIX/$SOL_PATH" 

	python3 main.py --seed $SEED2 solve \
		-i "$NTW_PREFIX/$NTW_NAME" \
		--objectives ${OBJECTIVES[*]} $REF_POINTS_OPT $REF_POINTS_STRING \
		--pop_size $POP_SIZE --n_gen $N_GEN \
		--n_replicas $N_REPLICAS \
		--n_partitions $N_PARTITIONS \
		--sampling_version $SAMPLING_VERSION \
		--crossover_version $CROSSOVER_VERSION \
		--mutation_version $MUTATION_VERSION \
		--mutation_prob_move $MUTATION_PROB_MOVE \
		--mutation_prob_change $MUTATION_PROB_CHANGE \
		--mutation_prob_binomial $MUTATION_PROB_BINOMIAL \
		--save_history \
		--algorithm $ALGORITHM \
		-o "$SOL_PREFIX/$SOL_PATH/$SOL_NAME"

}

hybrid_solve() {
	local VAR1=$1
	local SUFFIX="${VAR1:=}"

	local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"

    local SOL_FILES=()
    local SOL_FILES_GEN_LOAD_ORIG=()
    local SOL_FILES_GEN_LOAD_LAST=()
    local SOL_PATH_SUFFIX="hybrid_$(IFS='-'; echo "${HYBRID_ALGORITHMS[*]}")"
    for ALGORITHM in ${HYBRID_ALGORITHMS[*]}; do
        SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
        SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $HYBRID_POP_SIZE $HYBRID_N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL $SUFFIX)"
        SOL_FILES+=("$SOL_PREFIX/$SOL_PATH/$SOL_PATH_SUFFIX/$SOL_NAME")
        SOL_FILES_GEN_LOAD_ORIG+=("${SOL_FILES[-1]%.txt}_gen_load_orig.txt")
        SOL_FILES_GEN_LOAD_LAST+=("${SOL_FILES[-1]%.txt}_gen_load_last.txt")
    done

	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	mkdir -p "$SOL_PREFIX/$SOL_PATH/$SOL_PATH_SUFFIX" 

	python3 main.py --seed $SEED2 hybrid_solve \
		-i "$NTW_PREFIX/$NTW_NAME" \
		--objectives ${OBJECTIVES[*]} $REF_POINTS_OPT $REF_POINTS_STRING \
		--pop_size $HYBRID_POP_SIZE --n_gen $HYBRID_N_GEN \
        --gen_steps ${HYBRID_GEN_STEPS[*]} \
        --sol_dumping "${HYBRID_SOL_DUMPING//[[:blank:]]/}" \
		--n_replicas $N_REPLICAS \
		--n_partitions $N_PARTITIONS \
		--sampling_version $SAMPLING_VERSION \
		--crossover_version $CROSSOVER_VERSION \
		--mutation_version $MUTATION_VERSION \
		--mutation_prob_move $MUTATION_PROB_MOVE \
		--mutation_prob_change $MUTATION_PROB_CHANGE \
		--mutation_prob_binomial $MUTATION_PROB_BINOMIAL \
		--save_history \
		--algorithms ${HYBRID_ALGORITHMS[*]} \
		-o ${SOL_FILES[*]}
}

arrange() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		ALG_FILES[${SEED2}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
	done

	SOL_NAME="$(get_solution_filename $ALGORITHM "A" $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	python3 main.py arrange \
		--n_objectives $N_OBJECTIVES \
		-i ${ALG_FILES[*]} \
		-o "$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
}

arrange_all(){
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	i=1
	for ALG in ${ALGORITHMS[*]}; do
		SOL_NAME="$(get_solution_filename $ALG "A" $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		ALG_FILES[$i]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done

	SOL_NAME="$(get_solution_filename "ALL" "A" $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	python3 main.py arrange \
		--n_objectives $N_OBJECTIVES \
		-i ${ALG_FILES[*]} \
		-o "$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
}

solution_to_ref_points() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $REF_POINTS_ALGORITHM ${OBJECTIVES[@]})"
	SOL_NAME="$(get_solution_filename $REF_POINTS_ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	REF_FILENAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"

	mkdir -p "$SOL_PREFIX/$REF_PATH" 

	FACTOR=$1
	if [ -z ${FACTOR} ]; then
		local FACTOR_OPT=()
	else
		local FACTOR_OPT=(--lazy --lmb $FACTOR)
	fi

	python3 main.py --seed $SEED get_ref_points \
		--objectives ${OBJECTIVES[*]} \
		--ntw_file "$NTW_PREFIX/$NTW_NAME" \
		-i "$SOL_PREFIX/$SOL_PATH/$SOL_NAME" ${FACTOR_OPT[*]} \
		-o "$SOL_PREFIX/$REF_PATH/$REF_FILENAME"
}

solve_ilp() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS 'ILP' ${OBJECTIVES[@]})"

	rm -rf "$SOL_PREFIX/$SOL_PATH/*"

	mkdir -p "$SOL_PREFIX/$SOL_PATH/tmp"
	mkdir -p "$SOL_PREFIX/$SOL_PATH/log"

	i=1
	for l in ${LAMBDA_LIST[*]}; do
		# Async call
		{ time python3 main.py --seed $SEED2 solve \
			-i "$NTW_PREFIX/$NTW_NAME" \
			--algorithm "ILP" --n_partitions $N_PARTITIONS --single_mode --lmb $l \
			--n_replicas $N_REPLICAS \
			--verbose \
			--output "$SOL_PREFIX/$SOL_PATH/tmp/ref_$i"
		} &> "$SOL_PREFIX/$SOL_PATH/log/log_$i" &
		pids[${i}]=$!
		i=$((i+1))
	done

	i=1
	for pid in ${pids[*]}; do
		wait $pid
		if [ $? -eq 0 ]; then
			cat "$SOL_PREFIX/$SOL_PATH/tmp/ref_$i" | grep . >> "$SOL_PREFIX/$SOL_PATH/tmp.txt"
		fi
		i=$((i+1))
	done

	sort -f "$SOL_PREFIX/$SOL_PATH/tmp.txt" | uniq > "$SOL_PREFIX/$SOL_PATH/ILP.txt"
	rm "$SOL_PREFIX/$SOL_PATH/tmp.txt" 
}

plot_convergence() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
	SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"

	local MAX_GEN=$1
	if [ -z ${MAX_GEN} ]; then
		local MAX_GEN_OPT=()
	else
		local MAX_GEN_OPT=(--max_gen $MAX_GEN)
	fi
    
	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	python3 main.py --seed $SEED plot \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES ${REF_POINTS_}\
		-i "$SOL_PREFIX/$SOL_PATH/$SOL_NAME" \
		--history ${MAX_GEN_OPT[*]} \
		--title "Objective space - Convergence ($ALGORITHM) - $NODES:$TASKS:$USERS" \
		--trim_gen
}

plot_convergence_hybrid() {
	local MAX_GEN=$1
	if [ -z ${MAX_GEN} ]; then
		local MAX_GEN_OPT=()
	else
		local MAX_GEN_OPT=(--max_gen $MAX_GEN)
	fi

	local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
    local SOL_FILES=()
    local SOL_PATH_SUFFIX="hybrid_$(IFS='-'; echo "${HYBRID_ALGORITHMS[*]}")"
    for ALGORITHM in ${HYBRID_ALGORITHMS[*]}; do
        SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
        SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $HYBRID_POP_SIZE $HYBRID_N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
        SOL_FILES+=("$SOL_PREFIX/$SOL_PATH/$SOL_PATH_SUFFIX/$SOL_NAME")
    done

	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

    N_ALGS=${#HYBRID_ALGORITHMS[@]}
    for i in $(seq 0 1 $((N_ALGS-1))); do
        python3 main.py --seed $SEED plot \
            --objectives ${OBJECTIVES[*]} \
            --n_objectives $N_OBJECTIVES ${REF_POINTS_}\
            -i ${SOL_FILES[$i]} \
            --history ${MAX_GEN_OPT[*]} \
            --title "Objective space - Convergence (${HYBRID_ALGORITHMS[$i]}) - $NODES:$TASKS:$USERS" \
            --trim_gen &
 		pids[$i]=$!
    done

	for pid in ${pids[*]}; do
		wait $pid
	done

}

get_table_files_alg() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	TAB_FILES=()
	for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
		TAB_FILES[${SEED2}]="$ALY_PREFIX/$SOL_PATH/table_"$SEED2"_"$POP_SIZE
	done

	echo ${TAB_FILES[*]}
}

get_table_files_pop() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	TAB_FILES=()
	for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
		TAB_FILES[${SEED2}]="$ALY_PREFIX/$SOL_PATH/table_"$SEED2"_"$ALGORITHM
	done

	echo ${TAB_FILES[*]}
}

get_algorithm_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	i=0
	for ALG in ${ALGORITHMS[*]}; do
		SOL_NAME="$(get_solution_filename $ALG $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		ALG_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME" 
		i=$((i+1))
	done
	echo ${ALG_FILES[*]}
}

get_all_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	i=0
	for ALG in ${ALGORITHMS[*]}; do
		for SD in $(seq 1 1 $N_EXECUTIONS); do
			for POP in ${POP_SIZES[*]}; do
				SOL_NAME="$(get_solution_filename $ALG $SD $POP $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
				ALG_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME" 
				i=$((i+1))
			done
		done
	done
	echo ${ALG_FILES[*]}
}

get_operator_version_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	OP_FILES=()
	i=0
	for SV in ${SAMPLING_VERSION_LIST[*]}; do
	for CV in ${CROSSOVER_VERSION_LIST[*]}; do
	for MV in ${MUTATION_VERSION_LIST[*]}; do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SV $CV $MV $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		OP_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done
	done
	done

	echo ${OP_FILES[*]}
}

get_operator_version_legend() {
	OP_LEGEND=()
	i=0
	for SV in ${SAMPLING_VERSION_LIST[*]}; do
	for CV in ${CROSSOVER_VERSION_LIST[*]}; do
	for MV in ${MUTATION_VERSION_LIST[*]}; do
		OP_LEGEND[${i}]="SV$SV:CV$CV:MV$MV"
		i=$((i+1))
	done
	done
	done

	echo ${OP_LEGEND[*]}
}

get_mutation_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	MUT_FILES=()
	i=0
	for MM in ${MUTATION_PROB_MOVE_LIST[*]}; do
	for MC in ${MUTATION_PROB_CHANGE_LIST[*]}; do
	for MB in ${MUTATION_PROB_BINOMIAL_LIST[*]}; do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MM $MC $MB)"
		MUT_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done
	done
	done

	echo ${MUT_FILES[*]}
}

get_mutation_legend() {
	MUT_LEGEND=()
	i=0
	for MM in ${MUTATION_PROB_MOVE_LIST[*]}; do
	for MC in ${MUTATION_PROB_CHANGE_LIST[*]}; do
	for MB in ${MUTATION_PROB_BINOMIAL_LIST[*]}; do
		MUT_LEGEND[${i}]="MM$MM:MC$MC:MB$MB"
		i=$((i+1))
	done
	done
	done

	echo ${MUT_LEGEND[*]}
}

get_population_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	POP_FILES=()
	i=0
	for POP_SIZE in ${POP_SIZES[*]}; do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		POP_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done

	echo ${POP_FILES[*]}
}

get_population_legend() {
	POP_LEGEND=()
	i=0
	for POP_SIZE in ${POP_SIZES[*]}; do
		POP_LEGEND[${i}]="P$POP_SIZE"
		i=$((i+1))
	done

	echo ${POP_LEGEND[*]}
}

plot_comparison() {
	local VAR1=$1
	local INPUT="${VAR1:=algorithms}"
	local VAR2=$2
	local SUFFIX="${VAR2:=}"
	if [ $INPUT = "algorithms" ]; then
		FILES=$(get_algorithm_files)
		FILE_LEGEND=${ALGORITHMS[*]}
		TITLE="algorithms"
	elif [ $INPUT = "operator_versions" ]; then
		FILES=$(get_operator_version_files)
		FILE_LEGEND=$(get_operator_version_legend)
		TITLE="operator versions"
	elif [ $INPUT = "mutations" ]; then
		FILES=$(get_mutation_files)
		FILE_LEGEND=$(get_mutation_legend)
		TITLE="mutations"
	elif [ $INPUT = "population" ]; then
		FILES=$(get_population_files)
		FILE_LEGEND=$(get_population_legend)
		TITLE="populations"
	fi

	local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	python3 main.py --seed $SEED plot \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES $REF_POINTS_OPT $REF_POINTS_STRING \
		--ref_points_legend "$REF_POINTS_ALGORITHM RP" \
		-i $FILES \
		--comparison \
		--legend $FILE_LEGEND \
		--title "Objective space - Comparison between $TITLE $SUFFIX - $NODES:$TASKS:$USERS"
}

get_csv() {
	local VAR1=$1
	local SUFFIX="${VAR1:=}"
	FILES=$(get_all_files)
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	mkdir -p "$ALY_PREFIX/$SOL_PATH"
	python3 main.py analyze \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES $REF_POINTS_OPT $REF_POINTS_STRING \
		--network "$NTW_PREFIX/$NTW_NAME" \
		-i $FILES \
		--alg_names ${ALGORITHMS[*]} \
		--seed_values $(seq 1 1 $N_EXECUTIONS) \
		--pop_values ${POP_SIZES[*]} \
		--gen_values $(seq $GEN_STEP $GEN_STEP $N_GEN) \
		--output "$ALY_PREFIX/$SOL_PATH/table$SUFFIX.csv"
}

get_csv_hybrid() {
	local VAR1=$1
	local SUFFIX="${VAR1:=}"

    # Prepare all generation steps: step, step+1 and last generation
    NEW_TABLE_GEN_STEPS=()
    
    i=0; j=0
    while [ $i -lt ${#HYBRID_GEN_STEPS[*]} ] || [ $j -lt ${#HYBRID_TABLE_GEN_STEPS[*]} ]; do
        # Case where all HYBRID_GEN_STEPS have been processed
        if [ $i -eq ${#HYBRID_GEN_STEPS[*]} ]; then
            j_elem=${HYBRID_TABLE_GEN_STEPS[$j]}
            NEW_TABLE_GEN_STEPS+=($j_elem)
            j=$((j+1))

        # Case where all HYBRID_TABLE_GEN_STEPS have been processed
        elif [ $j -eq ${#HYBRID_TABLE_GEN_STEPS[*]} ]; then
            i_elem=${HYBRID_GEN_STEPS[$i]}
            i_next=$((i_elem + 1))
            NEW_TABLE_GEN_STEPS+=($i_elem $i_next)
            i=$((i+1))

        # Case where both arrays are still being processed
        else
            i_elem=${HYBRID_GEN_STEPS[$i]}
            i_next=$((i_elem + 1))
            j_elem=${HYBRID_TABLE_GEN_STEPS[$j]}
            if [ $j_elem -lt $i_elem ]; then
                NEW_TABLE_GEN_STEPS+=($j_elem)
                j=$((j+1))
            elif [ $i_elem == $j_elem ]; then
                NEW_TABLE_GEN_STEPS+=($i_elem $i_next)
                i=$((i+1))
                j=$((j+1))
                if [ ${HYBRID_TABLE_GEN_STEPS[$j]} -eq $i_next ]; then
                    j=$((j+1))
                fi
            elif [ $i_next == $j_elem ]; then
                NEW_TABLE_GEN_STEPS+=($i_elem $i_next)
                i=$((i+1))
                j=$((j+1))
            else
                NEW_TABLE_GEN_STEPS+=($i_elem $i_next)
                i=$((i+1))
            fi
        fi
    done

    # Get input filenames
    local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
    local SOL_FILES=()
    local SOL_PATH_SUFFIX="hybrid_$(IFS='-'; echo "${HYBRID_ALGORITHMS[*]}")"
    for ALGORITHM in ${HYBRID_ALGORITHMS[*]}; do
        for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
            local SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
            local SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $HYBRID_POP_SIZE $HYBRID_N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL $SUFFIX)"
            SOL_FILES+=("$SOL_PREFIX/$SOL_PATH/$SOL_PATH_SUFFIX/$SOL_NAME")
        done
    done

    # Reference points
	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	mkdir -p "$ALY_PREFIX/$SOL_PATH/$SOL_PATH_SUFFIX"
	python3 main.py analyze \
		--objectives ${OBJECTIVES[*]} \
        --entire_population \
		--n_objectives $N_OBJECTIVES $REF_POINTS_OPT $REF_POINTS_STRING \
        --n_algorithms ${#HYBRID_ALGORITHMS[*]} \
		--network "$NTW_PREFIX/$NTW_NAME" \
        -i ${SOL_FILES[*]} \
		--alg_names ${HYBRID_ALGORITHMS[*]} \
        --seed_values $(seq 1 1 $N_EXECUTIONS) \
		--pop_values $HYBRID_POP_SIZE \
		--gen_values ${NEW_TABLE_GEN_STEPS[*]} \
        --gen_steps ${HYBRID_GEN_STEPS[*]} \
		--output "$ALY_PREFIX/$SOL_PATH/$SOL_PATH_SUFFIX/table$SUFFIX.csv"
}

get_table() {
	local VAR1=$1
	local INPUT="${VAR1:=algorithms}"
	if [ $INPUT = "algorithms" ]; then
		FILES=$(get_algorithm_files)
		FILE_LEGEND=${ALGORITHMS[*]}
		suffix="$POP_SIZE"
	elif [ $INPUT = "population" ]; then
		FILES=$(get_population_files)
		FILE_LEGEND=$(get_population_legend)
		suffix="$ALGORITHM"
	fi

	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	mkdir -p "$ALY_PREFIX/$SOL_PATH"
	python3 main.py --seed $SEED analyze \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES $REF_POINTS_OPT $REF_POINTS_STRING \
		--gen_step 50 \
		-i $FILES \
		--alg_name $FILE_LEGEND \
		--network "$NTW_PREFIX/$NTW_NAME" \
		--output "$ALY_PREFIX/$SOL_PATH/table_"$SEED2"_$suffix"
}

get_table_group() {
	local VAR1=$1
	local INPUT="${VAR1:=algorithms}"
	if [ $INPUT = "algorithms" ]; then
		FILES=$(get_table_files_alg)
		suffix="$POP_SIZE"
	elif [ $INPUT = "population" ]; then
		FILES=$(get_table_files_pop)
		suffix="$ALGORITHM"
	fi

	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	output="$ALY_PREFIX/$SOL_PATH/table_M_$suffix"
	rm -f $output

	python3 cutre_table.py --col_start 2 --input ${FILES[*]} > $output
}

send_telegram_message() {
	# Send message using Telegram Bot API to notify that the process has finished
	ME=$(basename "$0")
	TOKEN=$(cat ../token.txt)
	CHAT_ID=$(cat ../chat_id.txt)
	HOSTNAME=$(hostname)
	curl -X POST -H 'Content-Type: application/json' \
		-d '{"chat_id": '$CHAT_ID', "text": "Script '$ME' has finished executing on server '$HOSTNAME'"}' \
		"https://api.telegram.org/bot$TOKEN/sendMessage"
	echo
}



