export N_CPUS=1
export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/branching_checkpoints
export PARENT_TEST_STATS_DIR=${PARENT_SAVE_DIR}/branching_test_stats/code_v1/random
export PARENT_INSTANCES_DIR=${PARENT_SAVE_DIR}/instances/mps
export PARENT_INSTANCES_METADATA_DIR=${PARENT_SAVE_DIR}/instances/metadata

export NTRAIN_INSTANCES=0
export NVAL_INSTANCES=0
export NTEST_INSTANCES=30
export SELECT_OPTION=first

export MODEL_DIFFICULTY=("all")
export MODEL_CODE_START_IDX=0
export MODEL_CODE_END_IDX=100
export MODEL_NTRAIN_INSTANCES=35
export MODEL_MAX_DATA_PER_INSTANCE=50
export MODEL_CODE_STR="code_v1"

export DIFFICULTY=("all")
export CODE_START_IDX=0
export CODE_END_IDX=100
export TIME_LIMIT=200
export CODE_STR="code_v1"
export TEST_INSTANCES_DIR_SPLIT_FILE=${PARENT_SAVE_DIR}/instances/metadata/by_dir_branching/code_v1/instances_split_ours+seed_EM.json

export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=3


CUDA_VISIBLE_DEVICES=0 python branching_test.py --n_cpus $N_CPUS --parent_model_dir $PARENT_MODEL_DIR --parent_test_stats_dir $PARENT_TEST_STATS_DIR \
        --parent_instances_dir $PARENT_INSTANCES_DIR --parent_instances_metadata_dir $PARENT_INSTANCES_METADATA_DIR \
        --ntrain_instances $NTRAIN_INSTANCES --nval_instances $NVAL_INSTANCES --ntest_instances $NTEST_INSTANCES --select_option $SELECT_OPTION \
        --model_difficulty "${MODEL_DIFFICULTY[@]}" --model_code_start_idx $MODEL_CODE_START_IDX --model_code_end_idx $MODEL_CODE_END_IDX \
        --model_ntrain_instances $MODEL_NTRAIN_INSTANCES --model_max_data_per_instance $MODEL_MAX_DATA_PER_INSTANCE \
        --model_code_str $MODEL_CODE_STR --code_str $CODE_STR --test_instances_dir_split_file $TEST_INSTANCES_DIR_SPLIT_FILE \
        --difficulty "${DIFFICULTY[@]}" --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX --time_limit $TIME_LIMIT \
        --emb_size $EMB_SIZE --edge_nfeats $EDGE_NFEATS --n_layers $N_LAYERS --not_test_by_dir --reevaluate 