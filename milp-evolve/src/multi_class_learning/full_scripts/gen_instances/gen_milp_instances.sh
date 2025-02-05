export N_CPUS=50
export N_INSTANCES=1000
export INSTANCES_SAVE_DIR=save_dir/instances
export CODE_DIR=milp_code_v1/code

export DIFFICULTY="all"
export CODE_START_IDX=0
export CODE_END_IDX=100
export VALID_TIMEOUT_DURATION=30
export TIMEOUT_DURATION=180
export SUBPROCESS_TIMEOUT_DURATION=240
export GAP_TH=0.0
export CODE_STR="code_v1"

python gen_instances.py --n_instances $N_INSTANCES --n_cpus $N_CPUS --instances_save_dir $INSTANCES_SAVE_DIR \
        --milp_type code --difficulty $DIFFICULTY --code_dir $CODE_DIR --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX \
        --valid_timeout_duration $VALID_TIMEOUT_DURATION --timeout_duration $TIMEOUT_DURATION \
        --subprocess_timeout_duration $SUBPROCESS_TIMEOUT_DURATION --not_optimize --gap_th $GAP_TH \
        --code_str $CODE_STR