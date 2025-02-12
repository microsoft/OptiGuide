export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/gap_checkpoints
export PARENT_TEST_STATS_DIR=${PARENT_SAVE_DIR}/gap_test_stats
export DATA_SPLIT_FILE=${PARENT_SAVE_DIR}/gap_data/metadata/code_v1/gap_data_split.json

export LABEL_CLIP_LB=0
export LABEL_CLIP_UB=1
export LABEL_MULT=100
export LABEL_KEY=lp_ip_gap

export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=1
export EVAL_SPLIT=test
export LOAD_STEP=30000
export CODE_STR="code_v1"

CUDA_VISIBLE_DEVICES=0 python gap_test.py --parent_model_dir $PARENT_MODEL_DIR --parent_test_stats_dir $PARENT_TEST_STATS_DIR \
        --label_clip_lb $LABEL_CLIP_LB --label_clip_ub $LABEL_CLIP_UB --label_mult $LABEL_MULT --label_key $LABEL_KEY \
        --model_load_data_split --model_data_split_file $DATA_SPLIT_FILE \
        --data_split_file $DATA_SPLIT_FILE --emb_size $EMB_SIZE --edge_nfeats $EDGE_NFEATS \
        --n_layers $N_LAYERS --code_str $CODE_STR --eval_split $EVAL_SPLIT --load_step $LOAD_STEP --reevaluate 