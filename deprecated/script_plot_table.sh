source script_constants.sh

python3 pd_tables.py \
    'data/analysis/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3-UNSGA3-SMSEMOA/table.csv' \
    'data/analysis/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3-UNSGA3-SMSEMOA/table_no_dumping.csv' \
    1 ${HYBRID_GEN_STEPS[*]} $HYBRID_N_GEN
