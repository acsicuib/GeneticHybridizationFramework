#!/bin/bash

source script_constants.sh
source script_functions.sh

FILE_LEGEND=${ALGORITHMS[*]}
TITLE="algorithms"

NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
if [ -f "$REF_FILE" ]; then
	REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
	REF_POINTS_OPT=--ref_points
else
	REF_POINTS_STRING=
	REF_POINTS_OPT=
fi

SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

PREFIX1="data/solutions/P$POP_SIZE-G$N_GEN/M$MUTATION_PROB/$NODES-$TASKS-$USERS"
PREFIX2="data/analysis/P$POP_SIZE-G$N_GEN/M$MUTATION_PROB/$NODES-$TASKS-$USERS"

mkdir -p "$ALY_PREFIX/$SOL_PATH"

for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
	FILES=$(get_algorithm_files)
	python3 main.py --seed $SEED analyze \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES $REF_POINTS_OPT $REF_POINTS_STRING \
		-i $FILES \
		--alg_name $FILE_LEGEND \
		--network "$NTW_PREFIX/$NTW_NAME" \
		--output "$ALY_PREFIX/$SOL_PATH/table_$SEED2"
done

output="$ALY_PREFIX/$SOL_PATH/table_M"
rm -f $output

# Print header (first row)
head -n1 "$ALY_PREFIX/$SOL_PATH/table_1" >> $output

for i in $(seq 2 1 $((N_ALGORITHMS+1))); do

	# Print each algorithm's name (first column)
	printf '%-12s' $(cat "$ALY_PREFIX/$SOL_PATH/table_1" | sed -n "${i}p" | awk '{print $1}') >> $output

	for j in {2..6}; do

		result=0.0

		for SEED2 in $(seq 1 1 $N_EXECUTIONS); do

			cell=$(cat "$ALY_PREFIX/$SOL_PATH/table_$SEED2" | sed -n "${i}p" | awk '{print $'$j'}')
			result=$(echo "scale=10; $result + $cell" | bc)

		done

		result=$(echo "scale=10; $result / 4" | bc)
		LC_ALL=C printf '%012.10f   ' $result >> $output

	done

	echo >> $output
done
