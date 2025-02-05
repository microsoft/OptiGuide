export PARENT_DATA_DIR=save_dir/contrast/data
export PARENT_INSTANCES_DIR=save_dir/instances/mps

python -u lang_extract_context.py --n_cpus $N_CPUS --parent_data_dir $PARENT_DATA_DIR --parent_instances_dir $PARENT_INSTANCES_DIR

export PARENT_CODE_DIR=milp_code_v1/code
export PARENT_INSTANCE_DIR=save_dir/instances/mps
export PARENT_OUTPUT_DIR=save_dir/contrast/conv

python contrast_mps_conv.py --parent_code_dir $PARENT_CODE_DIR --parent_instance_dir $PARENT_INSTANCE_DIR --parent_output_dir $PARENT_OUTPUT_DIR
