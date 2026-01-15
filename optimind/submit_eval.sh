#!/bin/bash
industryor="data/optimind_cleaned_classified_industryor.csv"
mamo_complex="data/optimind_cleaned_classified_mamo_complex.csv"
optmath="data/optimind_cleaned_classified_optmath.csv"

# This is only an illustrative subset of hints; you should provide a complete set of hints for each optimization class.
hint="data/hints_subset.csv"

qwen3_32b="Qwen/Qwen3-32B"

user_prompt_type=${1:-default}
system_prompt_type=${2:-default}


MAX_TURNS=5

TP_SIZE=4
DP_SIZE=2

TEMP=0.6
TOP_P=0.95

for num_majority in 1 8; do   
    for seed in 1 2 3 4 5 6 7 8 9 10; do
        echo "============================== Starting evaluations =============================="
        for data_name in  $industryor $mamo_complex $optmath ; do 
            echo "============================== Dataset: $data_name =============================="
            for model_path in $qwen3_32b ; do 
                echo "======================================= S G L A N G (NO HINTS, NO EARLY STOP) =========================================="
                echo "Running evaluation for dataset: $data_name"
                echo "Model path: $model_path"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 eval.py --data "$data_name" --backend_str sglang --model_path "$model_path"  --dp_size $DP_SIZE --tp_size $TP_SIZE --seed $seed \
                        --user_prompt_type $user_prompt_type --system_prompt_type $system_prompt_type \
                        --num-majority $num_majority --temp $TEMP --top-p $TOP_P  --debug --max-turns $MAX_TURNS --no_early_stop 
                echo "================================================================================================" 
                
                echo "======================================= S G L A N G (W/ HINTS + HINT INSTRUCTION, NO EARLY STOP) =========================================="
                echo "Running evaluation for dataset: $data_name"
                echo "Model path: $model_path"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 eval.py --data "$data_name" --backend_str sglang --model_path "$model_path"  --dp_size $DP_SIZE --tp_size $TP_SIZE --seed $seed \
                        --user_prompt_type $user_prompt_type --system_prompt_type $system_prompt_type \
                        --num-majority $num_majority --temp $TEMP --top-p $TOP_P \
                        --error_analysis_file $hint --apply_error_analysis --apply_hint_instructions  --debug --max-turns $MAX_TURNS --no_early_stop 
                echo "================================================================================================"
            done
        done
    done
done

