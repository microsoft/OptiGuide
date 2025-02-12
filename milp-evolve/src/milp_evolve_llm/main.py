import os
from src.evolve_batch import evolve_batch
from src.prefix_gen import gen_milp_code, get_formulation_methods, gen_topic_prefix
import argparse

parser = argparse.ArgumentParser(description="MILP Evolve Configuration")

parser.add_argument('--n_cpus', type=int, default=60, help="Number of CPUs for parameter search")
parser.add_argument('--resume', type=bool, default=True, help="Whether to resume from a checkpoint")
parser.add_argument('--output_dir', type=str, default="output_milp_v1", help="Directory to store output")

parser.add_argument('--llm_model_name', type=str, default="gpt-4o", help="Name of the model to use")
parser.add_argument('--max_prefix', type=int, default=2000, help="Maximum length of the prefix for the model")

# Evolve batch parameters
parser.add_argument('--max_select_per_depth', type=int, default=108, help="Maximum selections per depth")
parser.add_argument('--max_depth', type=int, default=2000, help="Maximum depth of the evolve search")
parser.add_argument('--n_sample_formulation_methods', type=int, default=3, help="Number of sample formulation methods")

# Timeout durations
parser.add_argument('--validity_timeout_duration', type=int, default=10, help="Timeout for validity checks")
parser.add_argument('--feas_timeout_duration', type=int, default=120, help="Timeout for feasibility checks")
parser.add_argument('--feas_subprocess_timeout_duration', type=int, default=180, help="Timeout for feasibility subprocess")

# Grid search parameters
parser.add_argument('--gs_timeout_duration', type=int, default=120, help="Grid search timeout duration")
parser.add_argument('--gs_subprocess_timeout_duration', type=int, default=180, help="Timeout for grid search subprocess")
parser.add_argument('--gs_n_round', type=int, default=1, help="Number of rounds for grid search")
parser.add_argument('--gs_n_samples_per_idx', type=int, default=60, help="Number of samples per index in grid search")
parser.add_argument('--gs_range_direction', type=str, default='mostly_inc_short', 
                    choices=["inc", "dec", "mostly_inc", "mostly_dec", "mostly_inc_short", "mostly_dec_short"], 
                    help="Direction of range for grid search")
parser.add_argument('--gs_final_time_th', type=int, default=20, help="Threshold for final grid search time")

# Load balancing parameters
parser.add_argument('--seed_wt', type=int, default=5, help="Weight for balancing seed files")
parser.add_argument('--evolve_wt', type=int, default=3, help="Weight for balancing evolve files")

args = parser.parse_args()


n_cpus = args.n_cpus
n_cpus = args.n_cpus  # number of cpus for parameter search
resume = args.resume
output_dir = args.output_dir

llm_model_name = args.llm_model_name
max_prefix = args.max_prefix

# evolve_batch parameters
max_select_per_depth = args.max_select_per_depth  
max_depth = args.max_depth
n_sample_formulation_methods = args.n_sample_formulation_methods


validity_timeout_duration = args.validity_timeout_duration
feas_timeout_duration = args.feas_timeout_duration
feas_subprocess_timeout_duration = args.feas_subprocess_timeout_duration

# grid search parameters
gs_timeout_duration = args.gs_timeout_duration
gs_subprocess_timeout_duration = args.gs_subprocess_timeout_duration 
gs_n_round = args.gs_n_round
gs_n_samples_per_idx = args.gs_n_samples_per_idx
gs_range_direction = args.gs_range_direction
gs_final_time_th = args.gs_final_time_th
seed_wt = args.seed_wt
evolve_wt = args.evolve_wt


remove_tmp = True  
save_satisfy = True  # whther to save the logs from all parameters that satisfy the requirements
fitness_criteria = ['avg_similarity'] 
PROMPTS_WEIGHTS_DICT = {"add": 0, "formulation_add": 1, "topic_add": 0.5, "conv_add": 0.5, "crossover_add": 1, 
                        "mutate": 1, "formulation_mutate": 0.8, "mutate_redundancy": 0.8, 
                        "topic_new": 1, "new": 0.8,  
                        "delete": 0.5} 


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    milp_list = gen_milp_code()

    topic_list = list(gen_topic_prefix())
    formulation_list = get_formulation_methods()
    out_name = os.path.join(output_dir, "milp_output.json")

    evolve_batch(milp_list, formulation_list, topic_list, 
            out_name, max_prefix=max_prefix, model_name=llm_model_name,
            n_cpus=n_cpus, n_sample_formulation_methods=n_sample_formulation_methods,
            max_depth=max_depth, max_select_per_depth=max_select_per_depth, 
            validity_timeout_duration=validity_timeout_duration,
            feas_timeout_duration=feas_timeout_duration,
            feas_subprocess_timeout_duration=feas_subprocess_timeout_duration, 
            gs_timeout_duration=gs_timeout_duration, 
            gs_subprocess_timeout_duration=gs_subprocess_timeout_duration,
            gs_n_round=gs_n_round, gs_n_samples_per_idx=gs_n_samples_per_idx, 
            gs_range_direction=gs_range_direction, gs_final_time_th=gs_final_time_th,
            remove_tmp=remove_tmp, save_satisfy=save_satisfy,
            resume=resume, fitness_criteria=fitness_criteria,
            PROMPTS_WEIGHTS_DICT=PROMPTS_WEIGHTS_DICT,
            seed_wt=seed_wt, evolve_wt=evolve_wt)

