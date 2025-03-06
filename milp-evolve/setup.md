# TOWARDS FOUNDATION MODELS FOR MIXED INTEGER LINEAR PROGRAMMING

This repository contains the code to our ICLR 2025 paper titled `TOWARDS FOUNDATION MODELS FOR MIXED INTEGER LINEAR PROGRAMMING`. 

This setup file provides instructions on environment setup, MILP-Evolve MILP class generation, as well as training and testing the three learning tasks.

## Environment Setup

Our implementation uses python 3.10, Pytorch 2.3.0, Pytorch Geometric 2.5.3 and Ecole 0.8.1 (SCIP 8.0.4, PySCIPOpt 4.3.0). The package dependencies are listed in `requirement.txt` and can be installed with the following command:

```
pip install -r requirements.txt
```

**Custom SCIP & PySCIPOpt Installation for Integrality Gap Prediction and Language-MILP Contrastive Learning:** To obtain the MILP input features for Integrality Gap Prediction and Language-MILP Contrastive Learning, we further use a custom version of the [SCIP](https://www.scipopt.org) solver (v7.0.2) and the [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) interface (v3.3.0), kindly provided by the authors of [1]. You can download the custom packages [here](https://www.dropbox.com/scl/fi/ei86qsmoxmiqviwi1g95b/scip_share.zip?rlkey=59x2o636t9njc5q8gjx5bw5dt&st=h9isjmcg&dl=0), and you should follow SCIP and PySCIPOpt instructions to build them from source. This custom envrionment allows us to extract the input feature at each LP relaxation. 

> [1] Paulus, Max B., et al. "Learning to cut by looking ahead: Cutting plane selection via imitation learning." International conference on machine learning. PMLR, 2022.

**Ecole, SCIP & PySCIPOpt Installation for Learning to Branch:** The learning to branch MILP input features are obtained from the [Ecole](https://github.com/ds4dm/ecole) library [2], which is built on SCIP (8.0.4) and PySCIPOpt (4.3.0) and allows us to extract the input feature at each B&B branching decision. Our `requirements.txt` install the Ecole library and the accompanied SCIP and PySCIPOpt by default.

> [2] Prouvost, Antoine et al. "Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers." Learning Meets Combinatorial Algorithms at NeurIPS2020, 2020. 

## MILP class generations with MILP-Evolve
We can run our LLM-based MILP Class Generation pipeline, MILP-Evolve, based on the following commands. One should also create a `.env` file inside the root folder (`./.env`) to provide OpenAI api authentication information (see `./milp_evolve_llm/src/util.py`)
```
cd ./milp_evolve_llm
N_CPUS=60   # Number of CPUs for parameter search and filtering
python main.py --n_cpus $N_CPUS
```
*A set of sample MILP Classes (seed and generated) are provided in `./sample_classes`*

We then copy the generated classes from `./milp_evolve_llm/output_milp_v1` to the `./multi_class_learning/milp_code_v1` folder.
```cd ./multi_class_learning
python gen_class_copy.py  
```
We then proceed in the `./multi_class_learning` folder to conduct all multi-class learning experiments.

## MILP instance generation from the MILP classes
First, we change the seed of each MILP class to generate `N_instances` MILP instances per class, for MILP classes whose index range from `CODE_START_IDX` to `CODE_END_IDX`. One can also checkout `full_scripts/gen_instances/gen_milp_instance.sh` for a expanded list of input parameters to run the code.

```script
export N_CPUS=50
export N_INSTANCES=1000    # number of instances per class
export CODE_START_IDX=0    # start index of the milp classes
export CODE_END_IDX=100    # end index of the milp classes
export CODE_DIR=milp_code_v1/code   # which directly the milp classes are in
export CODE_STR=code_v1   # the name of the generated instances directory

python gen_instances.py --n_cpus $N_CPUS --n_instances $N_INSTANCES --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX --code_dir $CODE_DIR --code_str $CODE_STR --not_optimize
```

## Integrality Gap Prediction

<details><summary><b>Data Collection</b></summary>

The integrality gap prediction data (MILP input features, integrality gap) can be obtained with the following code, which generates data for `NTRAIN_INSTANCES + NVAL_INSTANCES + NTEST_INSTANCES` MILP instances (in the following, 100) per MILP class within the range `CODE_START_IDX` and `CODE_END_IDX`. We improve a `TIME_LIMIT` when solving each MILP to obtain the integrality gap label. Instances that cannot be solved with a final gap `<0.1` will be removed from the learning experiments. One can also checkout `full_scripts/gap/collect_gap_data.sh` for a expanded list of input parameters to run the code.

```script
export N_CPUS=50
export NTRAIN_INSTANCES=70
export NVAL_INSTANCES=10
export NTEST_INSTANCES=20

export CODE_START_IDX=0
export CODE_END_IDX=100
export TIME_LIMIT=200
export CODE_STR="code_v1"

python gap_collect.py --n_cpus $N_CPUS --ntrain_instances $NTRAIN_INSTANCES --nval_instances $NVAL_INSTANCES --ntest_instances $NTEST_INSTANCES \
       --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX --time_limit $TIME_LIMIT --code_str $CODE_STR
```
</details>


<details>
<summary><b>Split all collected data into Train, Val, Test</b></summary>

Given the generated gap data, we split them into train, val, test using the following script, which generates a json file (by default, in `save_dir/gap_data/metadata/code_v1/gap_data_split.json`) with format `{"train": a list of training data path, "val": a list of validation data path, "test": a list of test data path}`.

```code
export CODE_STR="code_v1"
python gap_class_split.py --code_str $CODE_STR
```
</details>


<details>
<summary><b>Training</b></summary>

One can then use the following commands to train our integrality gap prediction model. One can also checkout `full_scripts/gap/train_gap.sh` for a expanded list of input parameters to run the code. 

```code
export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/gap_checkpoints   # location to save the training checkpoints
export PARENT_LOG_DIR=${PARENT_SAVE_DIR}/gap_logs   # location to save the training tensorboard logs
# location of the train/val/test data split json file; the saved model name is based the name of the json file
export DATA_SPLIT_FILE=${PARENT_SAVE_DIR}/gap_data/metadata/code_v1/gap_data_split.json    

export NB_EPOCHS=100   # maximum number of training epochs
export EVAL_EVERY=1000    # evaluate on the validation set per EVAL_EVERY gradient steps
export SAVE_EVERY=500    # save the checkpoint per SAVE_EVERY gradient steps
export TRAIN_BATCH_SIZE=16
export VAL_BATCH_SIZE=32

export LEARNING_RATE=0.001
export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=1
export CODE_STR="code_v1"

CUDA_VISIBLE_DEVICES=0 python gap_train.py --parent_model_dir $PARENT_MODEL_DIR 
         --parent_log_dir $PARENT_LOG_DIR --data_split_file $DATA_SPLIT_FILE \ 
         --nb_epochs $NB_EPOCHS --eval_every $EVAL_EVERY --save_every $SAVE_EVERY \
         --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE 
         --learning_rate $LEARNING_RATE --emb_size $EMB_SIZE --n_layers $N_LAYERS \
         --code_str $CODE_STR 
```
</details>

<details><summary><b> Testing </b> </summary>

One can then use the following commands to evaluate the integrality gap prediction model on the test set. One can also checkout `full_scripts/gap/test_gap.sh` for a expanded list of input parameters to run the code.

```script
export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/gap_checkpoints    # location of the the training checkpoints
export PARENT_TEST_STATS_DIR=${PARENT_SAVE_DIR}/gap_test_stats   # location to save the test stats
# location of the train/val/test data split json file
export DATA_SPLIT_FILE=${PARENT_SAVE_DIR}/gap_data/metadata/code_v1/gap_data_split.json    

export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=1
export EVAL_SPLIT=test
export LOAD_STEP=30000
export CODE_STR="code_v1"

CUDA_VISIBLE_DEVICES=0 python gap_test.py --parent_model_dir $PARENT_MODEL_DIR --parent_test_stats_dir $PARENT_TEST_STATS_DIR \
        --model_load_data_split --model_data_split_file $DATA_SPLIT_FILE \
        --data_split_file $DATA_SPLIT_FILE --emb_size $EMB_SIZE --edge_nfeats $EDGE_NFEATS \
        --n_layers $N_LAYERS --code_str $CODE_STR --eval_split $EVAL_SPLIT --load_step $LOAD_STEP --reevaluate 
```
</details>

## Learning to Branch
We adapt the imitation learning pipeline from the [Ecole](https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation/example.ipynb) library [2] to conduct the multi-class learning to branch experiment. 

> [2] Prouvost, Antoine et al. "Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers." Learning Meets Combinatorial Algorithms at NeurIPS2020, 2020. 

<details><summary><b> Data Collection </b></summary>

The branching data (MILP input features, strong branching action, candidate actions, strong branching score) at different B&B nodes per MILP instance can be obtained with the following code, which generates data for `NTRAIN_INSTANCES + NVAL_INSTANCES + NTEST_INSTANCES` MILP instances (in the following, 50) per MILP class within the range `CODE_START_IDX` and `CODE_END_IDX`. We improve a `TIME_LIMIT` when solving each MILP. At most `MAX_DATA_PER_INSTANCE` strong branching data at different B&B nodes can be collected per instance.
One can also checkout `full_scripts/branching/collect_branching_data.sh` for a expanded list of input parameters to run the code.

```script
export N_CPUS=50
export PARENT_SAVE_DIR=save_dir
export PARENT_DATA_DIR=${PARENT_SAVE_DIR}/branching_data   # location to save the branching data
export PARENT_DATA_METADATA_DIR=${PARENT_SAVE_DIR}/branching_data/metadata   # location to save the branching metadata
export PARENT_INSTANCES_DIR=${PARENT_SAVE_DIR}/instances/mps    # location where MILP instances are saved
export PARENT_INSTANCES_METADATA_DIR=${PARENT_SAVE_DIR}/instances/metadata    # location where MILP instances' metadata are saved

export MAX_DATA_PER_INSTANCE=50   # maximum number of B&B nodes per instance to collect the strong branching 
export NTRAIN_INSTANCES=35
export NVAL_INSTANCES=5 
export NTEST_INSTANCES=10

export CODE_START_IDX=0
export CODE_END_IDX=100  
export TIME_LIMIT=150
export CODE_STR="code_v1"

python branching_collect.py --n_cpus $N_CPUS --parent_data_dir $PARENT_DATA_DIR --parent_data_metadata_dir $PARENT_DATA_METADATA_DIR \
        --parent_instances_dir $PARENT_INSTANCES_DIR --parent_instances_metadata_dir $PARENT_INSTANCES_METADATA_DIR \
        --max_data_per_instance $MAX_DATA_PER_INSTANCE \
        --ntrain_instances $NTRAIN_INSTANCES --nval_instances $NVAL_INSTANCES --ntest_instances $NTEST_INSTANCES \
        --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX \
        --time_limit $TIME_LIMIT --code_str $CODE_STR
```
</details>

<details><summary><b> Split into Train, Val, Test </b></summary>

We then run the following code to split the MILP classes into disjoint train/val/test splits. We then take the branching data collected for the instances within the train classes to train the model. By default, this code produces json files such as `save_dir/instances/metadata/by_dir_branching/code_v1/instances_split_ours+seedEM.json` with format `{"train": a list of training MILP classes, "val": a list of validation MILP classes, "test": a list of testing MILP classes}`. One can modify the path information on top of this python file to save the json to a different location.

```script
export CODE_STR="code_v1"
export CODE_END_IDX=100
python branching_class_split.py --code_str $CODE_STR --code_end_idx $CODE_END_IDX
```
</details>

<details><summary><b> Training </b> </summary>

One can then use the following commands to train our branching model. One can also checkout `full_scripts/branching/train_branching.sh` for a expanded list of input parameters to run the code. 

```
export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/branching_checkpoints   # location to save the training checkpoints
export PARENT_LOG_DIR=${PARENT_SAVE_DIR}/branching_logs  # location to save the training tensorboard logs
export PARENT_DATA_DIR=${PARENT_SAVE_DIR}/branching_data   # location of the branching data 
export PARENT_INSTANCES_DIR=${PARENT_SAVE_DIR}/instances/mps   # # location of the MILP instances
export PARENT_INSTANCES_METADATA_DIR=${PARENT_SAVE_DIR}/instances/metadata   # location of the MILP instances' metadata

export TRAIN_BATCH_SIZE=32
export VAL_BATCH_SIZE=32

# location of the train/val/test MILP class split
export INSTANCES_DIR_SPLIT_FILE=${PARENT_SAVE_DIR}/instances/metadata/by_dir_branching/code_v1/instances_split_ours+seed_EM.json   
export CODE_STR="code_v1"
export CODE_START_IDX=0
export CODE_END_IDX=100

export NB_EPOCHS=300
export LEARNING_RATE=0.001
export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=3

CUDA_VISIBLE_DEVICES=0 python branching_train.py \
        --parent_model_dir $PARENT_MODEL_DIR --parent_log_dir $PARENT_LOG_DIR --parent_data_dir $PARENT_DATA_DIR --parent_instances_dir $PARENT_INSTANCES_DIR --parent_instances_metadata_dir $PARENT_INSTANCES_METADATA_DIR \
        --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE \
        --instances_dir_split_file $INSTANCES_DIR_SPLIT_FILE --code_str $CODE_STR --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX \
        --nb_epochs $NB_EPOCHS --emb_size $EMB_SIZE --learning_rate $LEARNING_RATE --edge_nfeats $EDGE_NFEATS --n_layers $N_LAYERS 
```
</details>

<details><summary><b> Testing </b> </summary>

One can then use the following commands to evaluate the branching model on the test set. One can also checkout `full_scripts/branching/branching_gap.sh` for a expanded list of input parameters to run the code.

```script
export N_CPUS=50   # number of cpus to evaluate different MILP instances in parallel
export PARENT_SAVE_DIR=save_dir
export PARENT_MODEL_DIR=${PARENT_SAVE_DIR}/branching_checkpoints   # location of the branching checkpoints
export PARENT_TEST_STATS_DIR=${PARENT_SAVE_DIR}/branching_test_stats/code_v1    # location to save the branching test stats
export PARENT_INSTANCES_DIR=${PARENT_SAVE_DIR}/instances/mps   # location of the MILP instances
export PARENT_INSTANCES_METADATA_DIR=${PARENT_SAVE_DIR}/instances/metadata   # location of the MILP instances' metadata

export NTEST_INSTANCES=50   # number of MILP instances to test per directory

# The following arguments define the model name
export MODEL_CODE_START_IDX=0
export MODEL_CODE_END_IDX=100
export MODEL_CODE_STR="code_v1"

# We filter out all test classes in TEST_INSTANCES_DIR_SPLIT_FILE whose idx is not within the following start and end values
export CODE_START_IDX=0   
export CODE_END_IDX=100
export TIME_LIMIT=200
export CODE_STR="code_v1"

# location of the train/val/test MILP class split
export TEST_INSTANCES_DIR_SPLIT_FILE=${PARENT_SAVE_DIR}/instances/metadata/by_dir_branching/code_v1/instances_split_ours+seed_EM.json

export EMB_SIZE=64
export EDGE_NFEATS=1
export N_LAYERS=3

CUDA_VISIBLE_DEVICES=0 python -u branching_test.py --n_cpus $N_CPUS \
        --parent_model_dir $PARENT_MODEL_DIR --parent_test_stats_dir $PARENT_TEST_STATS_DIR \
        --parent_instances_dir $PARENT_INSTANCES_DIR --parent_instances_metadata_dir $PARENT_INSTANCES_METADATA_DIR \
        --ntest_instances $NTEST_INSTANCES \
        --model_code_start_idx $MODEL_CODE_START_IDX --model_code_end_idx $MODEL_CODE_END_IDX \
        --model_code_str $MODEL_CODE_STR --code_str $CODE_STR \
        --test_instances_dir_split_file $TEST_INSTANCES_DIR_SPLIT_FILE \
        --code_start_idx $CODE_START_IDX --code_end_idx $CODE_END_IDX --time_limit $TIME_LIMIT \
        --emb_size $EMB_SIZE --edge_nfeats $EDGE_NFEATS --n_layers $N_LAYERS --not_test_by_dir --reevaluate 
```
</details>

## Language-MILP Contrastive Learning
<details><summary><b> Data Collection </b> </summary>

- Extract MILP input features: We run the following code to extract the MILP input features. The code is similar to `gap_collect.py`, except here we only solve each MILP instance to the root-node LP relaxation to collect the input feature, and we do not need to solve the MILP instance to optimal (only required to collect gap data).

```script
export N_CPUS=60
export PARENT_DATA_DIR=save_dir/contrast_data    # location to save the milp input features
export PARENT_INSTANCES_DIR=save_dir/instances/mps/code_v1   # location where the MILP instances are saved

python -u contrast_milp_collect.py --n_cpus $N_CPUS --parent_data_dir $PARENT_DATA_DIR --parent_instances_dir $PARENT_INSTANCES_DIR
```

- Generate Text Descriptions: We run the following code to generate the text descriptions based on the code and statistics extracted from the instances' A, b matrices.

```script
export PARENT_CODE_DIR=milp_code_v1/code   # location where the optimization code files are saved 
export PARENT_INSTANCE_DIR=save_dir/instances/mps/code_v1   # location where the MILP instances are saved
export PARENT_OUTPUT_DIR=save_dir/contrast/conv   # location to save the text descriptions

python contrast_mps_conv.py --parent_code_dir $PARENT_CODE_DIR --parent_instance_dir $PARENT_INSTANCE_DIR --parent_output_dir $PARENT_OUTPUT_DIR
```
</details>

<details><summary><b> Split into Train and Test Sets </b> </summary>

We then run the following code to split the multi-modal dataset (MILP and text) into disjoint train and test splits. In particular, `$MULTIMODAL_DATA_FILE` is a json file that contains a directory with format `[{"milp": path to the input features of the milp instance, "text_path": path to the text description of the milp instance}, ...]` to split into train and set sets with disjoint MILP classes. The train/test splits are saved as `{out_dir}/train_{out_suffix}_data.pkl.gz` and `{out_dir}/test_{out_suffix}_data.pkl.gz`, which are used to train and test the language-MILP contrastive model.

```
export PARENT_CODE_DIR=milp_code_v1/code
export PARENT_DATA_DIR=save_dir/contrast/data
export PARENT_DESC_DIR=save_dir/contrast/conv
export PARENT_SAVE_DIR=save_dir/contrast
export MULTIMODAL_DATA_FILE=save_dir/contrast/data_ours.json
export OUT_SUFFIX=ours_

python contrast_class_split.py --parent_code_dir  $PARENT_CODE_DIR --parent_data_dir $PARENT_DATA_DIR \
        --parent_desc_dir $PARENT_DESC_DIR --parent_save_dir $PARENT_SAVE_DIR \
        --multimodal_data_file $MULTIMODAL_DATA_FILE --out_suffix $OUT_SUFFIX
```

</details>

<details><summary><b> Training and Testing </b> </summary>

Finally, we can run the following command to train the language-MILP contrastive model.

```script
export DATASET=ours  
export EPOCH=1000
export EVAL_EPOCHS=10
export PRINT_ITERS=10000
export TEXT_TYPES="description only"

python contrast_train_test.py --epochs $EPOCH --dataset $DATASET --eval_epochs $EVAL_EPOCHS --print_iters $PRINT_ITERS --text_types $TEXT_TYPES
```
</details>


