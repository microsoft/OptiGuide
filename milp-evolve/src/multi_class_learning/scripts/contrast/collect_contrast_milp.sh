export N_CPUS=60
export PARENT_DATA_DIR=save_dir/contrast_data    # location to save the milp input features
export PARENT_INSTANCES_DIR=save_dir/instances/mps/code_v1   # location where the MILP instances are saved

python -u contrast_milp_collect.py --n_cpus $N_CPUS --parent_data_dir $PARENT_DATA_DIR --parent_instances_dir $PARENT_INSTANCES_DIR