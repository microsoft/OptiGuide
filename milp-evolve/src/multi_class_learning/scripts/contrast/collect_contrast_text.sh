export PARENT_CODE_DIR=milp_code_v1/code   # location where the optimization code files are saved 
export PARENT_INSTANCE_DIR=save_dir/instances/mps/code_v1   # location where the MILP instances are saved
export PARENT_OUTPUT_DIR=save_dir/contrast/conv   # location to save the text descriptions

python contrast_mps_conv.py --parent_code_dir $PARENT_CODE_DIR --parent_instance_dir $PARENT_INSTANCE_DIR --parent_output_dir $PARENT_OUTPUT_DIR