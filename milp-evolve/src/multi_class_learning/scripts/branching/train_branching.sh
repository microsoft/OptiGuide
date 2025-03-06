export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/branching_checkpoints
export PARENT_LOG_DIR=${PARENT_SAVE_DIR}/branching_logs
export PARENT_DATA_DIR=${PARENT_SAVE_DIR}/branching_data
export PARENT_INSTANCES_DIR=${PARENT_SAVE_DIR}/instances/mps
export PARENT_INSTANCES_METADATA_DIR=${PARENT_SAVE_DIR}/instances/metadata

export NB_EPOCHS=300
export EVAL_EVERY=1000
export SAVE_EVERY=500
export TRAIN_BATCH_SIZE=32
export VAL_BATCH_SIZE=32
export MAX_DATA_PER_INSTANCE=50

export NTRAIN_INSTANCES=35 
export NVAL_INSTANCES=5
export NTEST_INSTANCES=10 
export SELECT_OPTION=first
export NTRAIN_INSTANCES_DIR_VAL=70
export NVAL_INSTANCES_DIR_VAL=10
export NTEST_INSTANCES_DIR_VAL=20

export DIFFICULTY=("all") 
export CODE_START_IDX=0
export CODE_END_IDX=100
export INSTANCES_DIR_SPLIT_FILE=${PARENT_SAVE_DIR}/instances/metadata/by_dir_branching/code_v1/instances_split_ours+seed_EM.json

export LEARNING_RATE=0.001
export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=3
export CODE_STR="code_v1"

CUDA_VISIBLE_DEVICES=0 python branching_train.py --nb_epochs $NB_EPOCHS --eval_every $EVAL_EVERY --save_every $SAVE_EVERY \
        --max_data_per_instance $MAX_DATA_PER_INSTANCE --ntrain_instances $NTRAIN_INSTANCES --nval_instances $NVAL_INSTANCES --ntest_instances $NTEST_INSTANCES \
        --select_option $SELECT_OPTION --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE --learning_rate $LEARNING_RATE \
        --parent_model_dir $PARENT_MODEL_DIR --parent_log_dir $PARENT_LOG_DIR --parent_data_dir $PARENT_DATA_DIR --parent_instances_dir $PARENT_INSTANCES_DIR \
        --parent_instances_metadata_dir $PARENT_INSTANCES_METADATA_DIR \
        --difficulty "${DIFFICULTY[@]}" --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX  \
        --code_str $CODE_STR --instances_dir_split_file $INSTANCES_DIR_SPLIT_FILE --emb_size $EMB_SIZE --edge_nfeats $EDGE_NFEATS --n_layers $N_LAYERS \
        --ntrain_instances_dir_val $NTRAIN_INSTANCES_DIR_VAL --nval_instances_dir_val $NVAL_INSTANCES_DIR_VAL --ntest_instances_dir_val $NTEST_INSTANCES_DIR_VAL
