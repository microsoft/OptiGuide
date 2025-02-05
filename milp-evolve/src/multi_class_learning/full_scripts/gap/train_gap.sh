export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/gap_checkpoints
export PARENT_LOG_DIR=${PARENT_SAVE_DIR}/gap_logs
export DATA_SPLIT_FILE=${PARENT_SAVE_DIR}/gap_data/metadata/code_v1/gap_data_split.json

export LABEL_CLIP_LB=0
export LABEL_CLIP_UB=1
export LABEL_MULT=100
export LABEL_KEY=lp_ip_gap

export NB_EPOCHS=100
export EVAL_EVERY=1000
export SAVE_EVERY=500
export TRAIN_BATCH_SIZE=32
export VAL_BATCH_SIZE=32
export LEARNING_RATE=0.001

export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=1
export CODE_STR="code_v1"

CUDA_VISIBLE_DEVICES=0 python gap_train.py --parent_model_dir $PARENT_MODEL_DIR --parent_log_dir $PARENT_LOG_DIR \
        --label_clip_lb $LABEL_CLIP_LB --label_clip_ub $LABEL_CLIP_UB \
        --label_mult $LABEL_MULT --label_key $LABEL_KEY --nb_epochs $NB_EPOCHS --eval_every $EVAL_EVERY --save_every $SAVE_EVERY \
        --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE --learning_rate $LEARNING_RATE \
        --data_split_file $DATA_SPLIT_FILE --emb_size $EMB_SIZE --edge_nfeats $EDGE_NFEATS \
        --n_layers $N_LAYERS --code_str $CODE_STR
