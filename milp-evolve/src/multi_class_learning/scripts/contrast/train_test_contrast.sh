export EPOCH=1000
export DATASET=ours
export EVAL_EPOCHS=10
export PRINT_ITERS=10000
export TEXT_TYPES="description only"

python train.py --epochs $EPOCH --dataset $DATASET --eval_epochs $EVAL_EPOCHS --print_iters $PRINT_ITERS --text_types $TEXT_TYPES