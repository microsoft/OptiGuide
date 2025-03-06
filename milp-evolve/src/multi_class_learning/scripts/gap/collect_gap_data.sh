export N_CPUS=50
export PARENT_SAVE_DIR=save_dir
export PARENT_DATA_DIR=${PARENT_SAVE_DIR}/gap_data
export PARENT_DATA_METADATA_DIR=${PARENT_SAVE_DIR}/gap_data/metadata
export PARENT_INSTANCES_DIR=${PARENT_SAVE_DIR}/instances/mps
export PARENT_INSTANCES_METADATA_DIR=${PARENT_SAVE_DIR}/instances/metadata

export NTRAIN_INSTANCES=70
export NVAL_INSTANCES=10
export NTEST_INSTANCES=20
export SELECT_OPTION=first

export DIFFICULTY=("all")
export CODE_START_IDX=0
export CODE_END_IDX=100
export TIME_LIMIT=200
export CODE_STR="code_v1"

python gap_collect.py --n_cpus $N_CPUS --parent_data_dir $PARENT_DATA_DIR --parent_data_metadata_dir $PARENT_DATA_METADATA_DIR \
        --parent_instances_dir $PARENT_INSTANCES_DIR --parent_instances_metadata_dir $PARENT_INSTANCES_METADATA_DIR \
        --ntrain_instances $NTRAIN_INSTANCES --nval_instances $NVAL_INSTANCES --ntest_instances $NTEST_INSTANCES \
        --select_option $SELECT_OPTION --difficulty "${DIFFICULTY[@]}" --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX \
        --code_str $CODE_STR --time_limit $TIME_LIMIT