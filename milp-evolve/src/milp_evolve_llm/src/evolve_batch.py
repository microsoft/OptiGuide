import json
import os
import random
import time
import shutil
import numpy as np
import time
from collections import defaultdict
from src.utils import do_oai_reply, extract_code_block, insert_comments, check_valid_file, check_feasibility
from src.adjust_params import adjust_params_heuristics_search_i, detailed_parse_log
from src.utils import parallel_fn
from src.utils import parse_parameters_from_content, get_status_time_and_gap
from src.utils import parse_solving_process_from_content, reply_oai_embed

# Add prompts
from src.prompts.prompt_add import PROMPT1_A, SYSTEM_MESSAGE_A
from src.prompts.prompt_mutate import PROMPT1_M, SYSTEM_MESSAGE_M
from src.prompts.prompt_formulation_add import PROMPT1_FA, SYSTEM_MESSAGE_FA
from src.prompts.prompt_topic_add import PROMPT1_TA, SYSTEM_MESSAGE_TA
from src.prompts.prompt_topic_conv_add import PROMPT1_CONVA, SYSTEM_MESSAGE_CONVA
# Crossover prompt
from src.prompts.prompt_crossover_add import PROMPT1_CA, SYSTEM_MESSAGE_CA
# Mutate prompts
from src.prompts.prompt_formulation_mutate import PROMPT1_FM, SYSTEM_MESSAGE_FM
from src.prompts.prompt_redundancy_mutate import PROMPT1_RM, SYSTEM_MESSAGE_RM
# New prompts
from src.prompts.prompt_new import PROMPT1_N, SYSTEM_MESSAGE_N
from src.prompts.prompt_topic_new import PROMPT1_TN, SYSTEM_MESSAGE_TN
# Delete prompt
from src.prompts.prompt_delete import PROMPT1_D, SYSTEM_MESSAGE_D



EVOLVE_METHODS = ["add", "mutate", "new", "crossover_add", "formulation_add", "topic_add", "conv_add", 
                  "topic_new", "formulation_mutate", "mutate_redundancy", "delete"] 
EVOLVED_STR = ["add evolved", "mutate evolved", "new evolved", "crossover_add evolved", "formulation_add evolved", "topic_add evolved", "conv_add evolved", 
               "topic_new evolved", "formulation_mutate evolved", "mutate_redundancy evolved", "delete evolved"]  
EVOLVE_PROMPTS = [(SYSTEM_MESSAGE_A, PROMPT1_A), (SYSTEM_MESSAGE_M, PROMPT1_M), (SYSTEM_MESSAGE_N, PROMPT1_N), (SYSTEM_MESSAGE_CA, PROMPT1_CA), 
                  (SYSTEM_MESSAGE_FA, PROMPT1_FA), (SYSTEM_MESSAGE_TA, PROMPT1_TA), (SYSTEM_MESSAGE_CONVA, PROMPT1_CONVA), (SYSTEM_MESSAGE_TN, PROMPT1_TN),
                  (SYSTEM_MESSAGE_FM, PROMPT1_FM), (SYSTEM_MESSAGE_RM, PROMPT1_RM), (SYSTEM_MESSAGE_D, PROMPT1_D)] 

PROMPTS_WEIGHTS_DICT = {"add": 0, "formulation_add": 1, "topic_add": 0.5, "conv_add": 0.5, "crossover_add": 1, 
                        "mutate": 1, "formulation_mutate": 0.8, "mutate_redundancy": 0.8, 
                        "topic_new": 1, "new": 0.8, "delete": 0.5}  
               
def get_avail_evolve(avail_evolve_methods):
    evolve_methods = [EVOLVE_METHODS[i] for i in range(len(EVOLVE_METHODS)) if EVOLVE_METHODS[i] in avail_evolve_methods]
    evolved_str = [EVOLVED_STR[i] for i in range(len(EVOLVE_METHODS)) if EVOLVE_METHODS[i] in avail_evolve_methods]
    evolve_prompts = [EVOLVE_PROMPTS[i] for i in range(len(EVOLVE_METHODS)) if EVOLVE_METHODS[i] in avail_evolve_methods]
    return evolve_methods, evolved_str, evolve_prompts


def get_avail_evolve_methods(prefix, exclude_methods=[]):
    return [evolve for evolve in EVOLVE_METHODS if not prefix.get(evolve + " evolved", False) and evolve not in exclude_methods]


def sample_avail_evolve_methods(prefix, exclude_methods=[], evolve_methods=None, weight_dict=None):
    if evolve_methods == None:
        evolve_methods = EVOLVE_METHODS

    if weight_dict is None or not isinstance(weight_dict, dict):
        weight_dict = PROMPTS_WEIGHTS_DICT

    avail_evolve_methods = get_avail_evolve_methods(prefix, exclude_methods=exclude_methods)
    sample_evolve_methods = []
    for i, (evolve_method) in enumerate(evolve_methods):
        if evolve_method in avail_evolve_methods and np.random.rand() < weight_dict[evolve_method]:
            sample_evolve_methods.append(evolve_method)
    if len(sample_evolve_methods) == 0:
        sample_evolve_methods = [np.random.choice([evolve_method for evolve_method in avail_evolve_methods if weight_dict[evolve_method] > 0])]

    return sample_evolve_methods



def check_seed_feasibility(args):
    idx, filename, feas_timeout_duration, feas_subprocess_timeout_duration, verbose, remove_tmp = args
    print(f'Check code validity and feasibility for seed code {idx}:: {filename}...')
    feasible, solve_time, log = check_feasibility(filename, timeout_duration=feas_timeout_duration, 
                    subprocess_timeout_duration=feas_subprocess_timeout_duration, verbose=verbose)
    feasible_str = "feasible" if feasible else "infeasible"
    solve_time_str = 'None' if solve_time is None else f'{solve_time:.2f}'
    print(f'Finish checking seed code {idx}-{filename}:: {feasible_str}. Solve time {solve_time_str}') 
    # remove filename
    if remove_tmp and os.path.exists(filename):
        os.remove(filename)
    return feasible, solve_time, log


def check_evolve_feasbility(args):
    code, tmp_dir, tmp_idx, validity_timeout_duration, feas_timeout_duration, feas_subprocess_timeout_duration, remove_tmp = args

    print(f'Check code validity and feasibility for code of batch idx {tmp_idx}...')

    # check code validity by writing it in a file and call the file
    tmp_filename = os.path.join(tmp_dir, f"milp_{tmp_idx}.py")
    if not os.path.exists(tmp_filename):
        open(tmp_filename, "w").write(code)      

    ret_feasible, ret_criteria_met, ret_solve_time, ret_log = False, False, None, ""
    if not check_valid_file(tmp_filename, timeout_duration=validity_timeout_duration): 
        print(f"Invalid code of id {tmp_idx}. remove")
    else:
        # check feasibility and solve time: skip if infeasible
        ret_feasible, ret_solve_time, ret_log = check_feasibility(tmp_filename, timeout_duration=feas_timeout_duration, 
                                subprocess_timeout_duration=feas_subprocess_timeout_duration, verbose=True)
        if not ret_feasible:  # TODO: maybe relax this in the future: only remove if parameter search is unsuccessful too
            print(f"Infeasible code of id {tmp_idx}. remove")
        else:
            ret_criteria_met = detailed_parse_log(ret_log)

    if remove_tmp:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
    else:
        # keep milp and write log
        if not os.path.exists(tmp_filename):
            open(tmp_filename, "w").write(code)
        tmp_log_filename = os.path.join(tmp_dir, f"log_{tmp_idx}.txt")
        open(tmp_log_filename, "w").write(ret_log)

    
    feasible_str = "feasible" if ret_feasible else "infeasible"
    criteria_met_str = "criteria met" if ret_criteria_met else "criteria not met"
    solve_time_str = 'None' if ret_solve_time is None else f'{ret_solve_time:.2f}'

    print(f'Finish checking code of batch idx {tmp_idx} ({tmp_filename}):: {feasible_str} | {criteria_met_str}. Solve time {solve_time_str}...')

    return ret_feasible, code, ret_solve_time, ret_log


def get_fitness_score(code_embedding, log_data_list, prefixes, detailed_logs=False, fitness_criteria="all"):
    def cosine_similarity(embedding1, embedding2):
        # Compute cosine similarity between two embeddings
        dot_product = sum(a*b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a*a for a in embedding1) ** 0.5
        magnitude2 = sum(b*b for b in embedding2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def load_embedding_from_path(path):
        if os.path.exists(path):
            with open(path, 'r') as file:
                embedding = json.load(file)
            return embedding
        return None

    def calculate_diversity_score(new_value, previous_values, operation='max', top_k=5):
        if not previous_values:
            return 1  # Max diversity if no previous values

        if operation == 'max':
            max_distance = max(abs(new_value - pv) for pv in previous_values)
            return max_distance / (max_distance + 1)

        elif operation == 'avg':
            avg_distance = sum(abs(new_value - pv) for pv in previous_values) / len(previous_values)
            return avg_distance / (avg_distance + 1)
        
        elif operation == 'top5':
            sorted_distances = sorted(abs(new_value - pv) for pv in previous_values)
            top5_distances = sorted_distances[:top_k]
            avg_top5_distance = sum(top5_distances) / len(top5_distances) if top5_distances else 0
            return avg_top5_distance / (avg_top5_distance + 1)

    # if fitness_criteria is string, we will change it to list
    if isinstance(fitness_criteria, str):
        fitness_criteria = [fitness_criteria]
        
    # Parse the solving process log to extract stats
    fitness_features = []
    fitness_score = 0
    
    # Three different difficulty levels, may or may not exist
    for log_data in log_data_list:
        if log_data.strip() == "": continue
        
        stats = parse_solving_process_from_content(log_data, detailed_logs=detailed_logs)
        # Unpack relevant stats
        total_solving_time = float('inf') if stats["Total Solving Time"] is None else stats["Total Solving Time"]
        total_vars = 0 if stats["Total Variables"] is None else stats["Total Variables"]
        binary_vars = 0 if stats["Binary Variables"] is None else stats["Binary Variables"]
        integer_vars = 0 if stats["Integer Variables"] is None else stats["Integer Variables"]
        total_cons = 0 if stats["Total Constraints"] is None else stats["Total Constraints"]
        num_bnb_nodes = 0 if stats["Number of BnB Nodes"] is None else stats["Number of BnB Nodes"]
        frac_bin_int_vars = 0 if total_vars == 0 else (binary_vars + integer_vars) / total_vars
        total_binint_vars = binary_vars + integer_vars
        integrality_gap = float('inf') if stats["Integrality Gap"] is None else stats["Integrality Gap"]

        if stats["Gap"] is None:
            continue

        # Collect fitness features
        feature_dict = {
            "total_solving_time": total_solving_time,
            "num_bnb_nodes": num_bnb_nodes,
            "total_vars": total_vars,
            "total_binint_vars": total_binint_vars,
            "total_cons": total_cons,
            "frac_binint_vars": frac_bin_int_vars,
            "integrality_gap": integrality_gap
        }
        fitness_features.append(feature_dict)

    if not fitness_features:
        return False, None, 0

    total_solving_time = sum([feature["total_solving_time"] for feature in fitness_features]) / len(fitness_features)
    num_bnb_nodes = sum([feature["num_bnb_nodes"] for feature in fitness_features]) / len(fitness_features)
    total_vars = sum([feature["total_vars"] for feature in fitness_features]) / len(fitness_features)
    total_binint_vars = sum([feature["total_binint_vars"] for feature in fitness_features]) / len(fitness_features)
    total_cons = sum([feature["total_cons"] for feature in fitness_features]) / len(fitness_features)
    frac_binint_vars = sum([feature["frac_binint_vars"] for feature in fitness_features]) / len(fitness_features)
    integrality_gap = sum([feature["integrality_gap"] for feature in fitness_features]) / len(fitness_features)
                                                                                                  
    fitness_features = {
        "total_solving_time": total_solving_time,
        "num_bnb_nodes": num_bnb_nodes,
        "total_vars": total_vars,
        "total_binint_vars": total_binint_vars,
        "total_cons": total_cons,
        "frac_binint_vars": frac_bin_int_vars,
        "integrality_gap": integrality_gap
    }

    if not prefixes:
        return True, fitness_features, 1
    
    # Load previous embeddings from the specified paths
    previous_embeddings = []
    for prefix in prefixes:
        embedding_path = prefix["embedding"]
        prev_embedding = load_embedding_from_path(embedding_path)
        if prev_embedding is not None:
            previous_embeddings.append(prev_embedding)

    # Calculate similarity score with previous embeddings
    if code_embedding is None:
        similarity_scores = []
    else:
        similarity_scores = [cosine_similarity(code_embedding, prev_embedding) for prev_embedding in previous_embeddings]
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 1  # if no previous embedding, set to None

    # get previous values if exists, else do 0
    previous_solving_times = [getattr(prefix["fitness_features"], "total_solving_time", 0) for prefix in prefixes]
    previous_bnb_nodes = [getattr(prefix["fitness_features"], "num_bnb_nodes", 0) for prefix in prefixes]
    previous_total_vars = [getattr(prefix["fitness_features"], "total_vars", 0) for prefix in prefixes]
    previous_total_binint_vars = [getattr(prefix["fitness_features"], "total_binint_vars", 0) for prefix in prefixes]
    previous_total_cons = [getattr(prefix["fitness_features"], "total_cons", 0) for prefix in prefixes]
    previous_frac_binint_vars = [getattr(prefix["fitness_features"], "frac_binint_vars", 0) for prefix in prefixes]
    previous_integrality_gaps = [getattr(prefix["fitness_features"], "integrality_gap", 0) for prefix in prefixes]


    solving_time_diversity_score = calculate_diversity_score(total_solving_time, previous_solving_times)
    bnb_nodes_diversity_score = calculate_diversity_score(num_bnb_nodes, previous_bnb_nodes)
    variable_count_diversity_score = calculate_diversity_score(total_vars, previous_total_vars)
    binint_variable_count_diversity_score = calculate_diversity_score(total_binint_vars, previous_total_binint_vars)
    constraint_count_diversity_score = calculate_diversity_score(total_cons, previous_total_cons)
    frac_binint_vars_diversity_score = calculate_diversity_score(frac_binint_vars, previous_frac_binint_vars)
    integrality_gap_diversity_score = calculate_diversity_score(integrality_gap, previous_integrality_gaps)

    fitness_score = []
    if "total_solving_time" in fitness_criteria or "all" in fitness_criteria:
        fitness_score.append(solving_time_diversity_score)
    elif "bnb_nodes" in fitness_criteria or "all" in fitness_criteria:
        fitness_score.append(bnb_nodes_diversity_score)
    elif "variable_count" in fitness_criteria or "all" in fitness_criteria:
        fitness_score.append(variable_count_diversity_score)
    elif "binint_variable_count" in fitness_criteria:   # if all: only include frac_bin_int_vars
        fitness_score.append(binint_variable_count_diversity_score)
    elif "constraint_count" in fitness_criteria or "all" in fitness_criteria:
        fitness_score.append(constraint_count_diversity_score)
    elif "frac_binint_vars" in fitness_criteria or "all" in fitness_criteria:
        fitness_score.append(frac_binint_vars_diversity_score)
    elif "integrality_gap" in fitness_criteria or "all" in fitness_criteria:
        fitness_score.append(integrality_gap_diversity_score)
    elif "avg_similarity" in fitness_criteria or "all" in fitness_criteria and similarity_scores:  # if embedding exists
        fitness_score.append(1 - avg_similarity_score)
    
    if len(fitness_score) == 0:
        # code similarity as the diversity score
        fitness_score.append(1 - avg_similarity_score)

    # pdb.set_trace()
    fitness_score = sum(fitness_score) / len(fitness_score)
    return True, fitness_features, fitness_score


def load_generated_from_depth_i(i_depth, prefixes):
    generated = set()
    for prefix in prefixes:
        if prefix["depth"] == i_depth:
            system_message = open(prefix["system_message"], "r").read()
            prompt = open(prefix["prompt"], "r").read()
            prompt_ = '\n'.join([system_message, prompt])
            generated.add(prompt_)
    return generated

    
def get_prefixes_depth(prefixes, prev_depth, fail_depth, fail_th=3, restart_depth_th=5):
    prefixes_depth = [prefix for prefix in prefixes if prefix["depth"] == prev_depth]
    if fail_depth > fail_th and prev_depth > 0:
        depth_go_back = max(prev_depth - (fail_depth - fail_th), 0)
        print(f'Failing {fail_depth} > {fail_th} times. Add all prefixes from {depth_go_back} '
                f'to {prev_depth} to generate prompts ...')
        for i_depth in range(prev_depth, depth_go_back-1, -1):
            prefixes_depth += [prefix for prefix in prefixes if prefix["depth"] == i_depth]
        prev_depth = depth_go_back
        
    while prev_depth > 0 and len(prefixes_depth) <= restart_depth_th:
        if prev_depth < 0:
            print(f'No more prefixes to evolve. Exit.')
            break
        print(f'Too few ({len(prefixes_depth)}) prefixes at depth {prev_depth} to evolve...')
        prev_depth -= 1
        print(f'Go back to depth {prev_depth} to generate more prefixes ...')
        prefixes_depth = [prefix for prefix in prefixes if prefix["depth"] == prev_depth]
    return prefixes_depth


def gen_prompts_depth(prefixes, cur_depth, prefixes_depth, depth_data_path, 
                      formulation_list, topic_list,  
                      n_sample_formulation_methods=1, max_select_per_depth=1000, 
                      resume=True,
                      solve_time_high=150, repeat_th=5, n_repeat_sample=3, 
                      PROMPTS_WEIGHTS_DICT=PROMPTS_WEIGHTS_DICT,
                      seed_wt=0, evolve_wt=0, include_seed_depth=10):
    # load from previously generated  at the current depth (in case system restart)
    generated = load_generated_from_depth_i(cur_depth, prefixes)
    # load the prompts that are supposed to be run at the current depth, if exists
    if resume and os.path.exists(depth_data_path):
        prompt_list, new_prefix_list = [], []
        depth_data = json.load(open(depth_data_path, 'r'))
        # filter out all generated (depth_data, prompt_list, new_prefix_list)
        prompt_list_save = depth_data['prompt_list']
        for i, prompt_ in enumerate(prompt_list_save):
            if prompt_ in generated:
                continue
            prompt_list.append(prompt_)
            new_prefix_list.append(depth_data['new_prefix_list'][i])
        print(f'Load depth data from {depth_data_path} with {len(prompt_list)} prompts to do ({len(prompt_list_save)} prompts in total)...')
    else:
        prompt_list, new_prefix_list, fitness_score_list = [], [], []
        
        seed_prefix_count_dict = defaultdict(int) # how many times the root has been evolved (promote load balance)
        evolve_count_dict = defaultdict(int)  # how many times the prompt has been used (promote load balance)
        # for all prefixes: count how many times the root has been evolved
        prefix_root_idxs = defaultdict(int)
        
        for idx, prefix in enumerate(prefixes):
            if prefix["root"] == prefix["id"]:
                prefix_root_idxs[prefix["root"]] = idx
            seed_prefix_count_dict[prefix["root"]] += 1
            evolve_count_dict[prefix["evolve"]] += 1

        num_prefixes_depth = len(prefixes_depth)
        # for the current level: count how many times seed prefix appears
        seed_prefix_count_depth = defaultdict(int)
        for prefix in prefixes_depth:
            seed_prefix_count_depth[prefix["root"]] += 1
        
        # add more seed prefixes if the seed prefix has not been evolved enough
        for root in seed_prefix_count_dict:
            if root not in seed_prefix_count_depth or seed_prefix_count_depth[root] < 2:  #  or (cur_depth > 0 and cur_depth % include_seed_depth == 0)
                # add the seed file for this root to generate more
                prefixes_depth.append(prefixes[prefix_root_idxs[root]])
        print(f'Update prefixes_depth # from {num_prefixes_depth} to {len(prefixes_depth)}...')
    
        # normalize by sum count
        sum_prefix_count = sum(seed_prefix_count_dict.values())
        sum_evolve_count = sum(evolve_count_dict.values())
        if sum_prefix_count == 0 or sum_evolve_count == 0:
            seed_prefix_score_dict = {k: 1 for k in seed_prefix_count_dict}
            evolve_score_dict = {k: 1 for k in evolve_count_dict}
        else:
            seed_prefix_score_dict = {k: 1 - v / sum_prefix_count for k, v in seed_prefix_count_dict.items()}        
            evolve_score_dict = {k: 1 - v / sum_evolve_count for k, v in evolve_count_dict.items()}
        
        for prefix in prefixes_depth:
            exclude_methods = []
            if cur_depth < 2:
                exclude_methods.append('crossover_add')
                exclude_methods.append('delete')

            # if most of param search parent samples are 180s then we should set a higher weight for delete and reduce weight for add
            if "solve_time" in prefix and prefix["solve_time"] > solve_time_high:
                prompts_weights_dict = PROMPTS_WEIGHTS_DICT.copy()
                for key in prompts_weights_dict:
                    # add more weight to delete and reduce weight for add
                    if key == 'delete':
                        prompts_weights_dict[key] = 1  # max(prompts_weights_dict[key], 0.8)
                    elif 'add' in key:
                        prompts_weights_dict[key] = max(prompts_weights_dict[key] * 0.5, 0)
            else:
                prompts_weights_dict = PROMPTS_WEIGHTS_DICT

            # evolve_methods = get_avail_evolve_methods(prefix, exclude_methods=exclude_methods)
            evolve_methods = sample_avail_evolve_methods(prefix, exclude_methods=exclude_methods, weight_dict=prompts_weights_dict)
            evolve_methods, evolve_str, evolve_prompts = get_avail_evolve(evolve_methods)
            
            if len(prefixes_depth) < repeat_th:
                for _ in range(n_repeat_sample-1):
                    evolve_methods_ = sample_avail_evolve_methods(prefix, exclude_methods=exclude_methods, weight_dict=prompts_weights_dict)
                    evolve_methods_, evolve_str_, evolve_prompts_ = get_avail_evolve(evolve_methods_)
                    evolve_methods += evolve_methods_
                    evolve_str += evolve_str_
                    evolve_prompts += evolve_prompts_
            
            original_code = open(prefix["milp"], "r").read()
            original_code_with_comment = insert_comments(original_code)  # add comment to modify the code

            for evolve, (system_message, PROMPT) in zip(evolve_methods, evolve_prompts):
                if evolve == "crossover_add":
                    # randomly choose a id, read the code from path
                    second_code = open(random.choice([x["milp"] for x in prefixes]), "r").read()
                    prompt = PROMPT.format(code=original_code_with_comment, code2=second_code) 
    
                elif evolve == "formulation_add" or evolve == "formulation_mutate":
                    sampled_formulation_methods = np.random.choice(formulation_list, n_sample_formulation_methods, replace=False)
                    prompt = PROMPT.format(formulation_methods=sampled_formulation_methods, code=original_code_with_comment)
                elif evolve == "topic_add" or evolve == "topic_new":
                    sample_topic = random.choice(topic_list)
                    prompt = PROMPT.format(topic=sample_topic, code=original_code_with_comment)
                elif evolve == "conv_add":
                    sample_capital_letters = random.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", random.randint(3, 5))
                    sample_topic = random.choice(topic_list)
                    prompt = PROMPT.format(capital_letters=sample_capital_letters, capital_letters2=sample_capital_letters,
                                        topic=sample_topic, code=original_code_with_comment)
                else:
                    assert evolve in ["add", "mutate","new", "mutate_redundancy", "delete", "delete_obj"], f"Unknown evolve method {evolve}"
                    prompt = PROMPT.format(code=original_code_with_comment)  
                
                prompt_ = '\n'.join([system_message, prompt])
                if prompt_ in generated:
                    continue
                generated.add(prompt_)

                new_prefix_list.append((system_message, prompt, evolve, prefix["id"]))
                prompt_list.append(prompt_)

                fitness_score = getattr(prefix, "fitness_score", 1)
                seed_prefix_score = seed_prefix_score_dict[prefix["root"]]
                evolve_score = evolve_score_dict[prefix["evolve"]]

                fitness_score_list.append(fitness_score + seed_wt * seed_prefix_score + evolve_wt * evolve_score)
        
        # normalize fitness_score_list with numpy
        fitness_p = np.array(fitness_score_list) / sum(fitness_score_list)
        subsample_idxs = np.random.choice(len(prompt_list), min(len(prompt_list), max_select_per_depth), 
                                          p=fitness_p, replace=False)
        new_prefix_list = [new_prefix_list[i] for i in subsample_idxs] 
        prompt_list = [prompt_list[i] for i in subsample_idxs] 

        # Save all prompt_list to do from the current depth for restarting purpose
        os.makedirs(os.path.dirname(depth_data_path), exist_ok=True)
        depth_data = {'prompt_list': prompt_list, 
                      'new_prefix_list': new_prefix_list}
        open(depth_data_path, 'w').write(json.dumps(depth_data, indent=2))

    return prompt_list, new_prefix_list



def param_search_i(tmp_idx, param_search_dir_i, code, log, gs_n_samples_per_idx, gs_timeout_duration,
                    gs_subprocess_timeout_duration, gs_range_direction, gs_n_round, gs_final_time_th, n_cpus, 
                    remove_tmp, save_tmp_file_path_i, save_satisfy, save_satisfy_file_path_i):
    # save the code to the parameter search directory
    save_file_path = os.path.join(param_search_dir_i, f"milp.py")
    save_log_path = os.path.join(param_search_dir_i, f"log.txt")
    open(save_file_path, "w").write(code)
    open(save_log_path, "w").write(log)

    org_params = parse_parameters_from_content(code)
    org_criteria_met = detailed_parse_log(log)
    status, solve_time, gap = get_status_time_and_gap(log)

    print(f'\nSTART PARAMETER SEARCH FOR CODE OF ID {tmp_idx}:: ORG PARMAETERS {org_params}, STATUS {status} SOLVE TIME {solve_time} GAP {gap} CRITERIA MET {org_criteria_met}...')
    success = adjust_params_heuristics_search_i(org_params, solve_time, save_file_path, param_search_dir_i, 
                                n_samples_per_idx=gs_n_samples_per_idx, timeout_duration=gs_timeout_duration, 
                                subprocess_timeout_duration=gs_subprocess_timeout_duration, range_direction=gs_range_direction, 
                                n_round=gs_n_round, n_cpus=n_cpus, final_time_th=gs_final_time_th, remove_tmp=remove_tmp,
                                save_tmp_file_path=save_tmp_file_path_i, save_satisfy=save_satisfy, save_satisfy_file_path=save_satisfy_file_path_i)
    if not success:
        if not org_criteria_met:
            print(f"Parameter search failed for code of id {tmp_idx} and original parameter does not meet criteria.")
            return False, code, log
        else:
            print(f"Parameter search failed for code of id {tmp_idx} but original parameter meets criteria. Keep the parameter search folder.")
            difficulty = "hard" if solve_time > gs_timeout_duration * 2 / 3 else "medium" if solve_time > gs_timeout_duration * 1 / 3 else "easy"
            # save code to new_milp.py and new_milp_{difficulty}.py
            for from_data, to_name in [(code, "new_milp.py"), (code, f"new_milp_{difficulty}.py"), (log, "new_log.txt"), (log, f"new_log_{difficulty}.txt")]:
                open(os.path.join(param_search_dir_i, to_name), "w").write(from_data)
    else:
        new_file_path = os.path.join(param_search_dir_i, f"new_milp.py")
        new_log_path = os.path.join(param_search_dir_i, f"new_log.txt")
        code = open(new_file_path, "r").read()
        log = open(new_log_path, "r").read()
    
    return True, code, log



def save_data_i(code, log, system_message, prompt, ans, code_embedding, 
                prefixes, evolve, id_from, cur_depth, param_search_dir_i, 
                model_name, fitness_criteria, codedir, details_outdir, out_name, last_cache_time):
    # logs of all three difficulty levels
    log_data_list = []
    for difficulty in ["easy", "medium", "hard"]:
        log_file = os.path.join(param_search_dir_i, f"new_log_{difficulty}.txt")
        if os.path.exists(log_file):
            log_data_list.append(open(log_file, "r").read())

    success, fitness_features, fitness_score = get_fitness_score(code_embedding, log_data_list, prefixes, fitness_criteria=fitness_criteria)
    if not success:
        print(f"Fail to get fitness score for code of id {len(prefixes)} ... Remove this code")
        return False, prefixes, last_cache_time

    _, solve_time, _ = get_status_time_and_gap(log)

    prefix_id = len(prefixes)
    prefixes.append(
        {
            "milp": code,
            "id": prefix_id,
            "source": model_name,
            "evolve": evolve,
            "evolve from": id_from,
            "root": prefixes[id_from]["root"],
            "depth": cur_depth,
            "system_message": system_message,
            "prompt": prompt, 
            "output": ans, 
            "log": log,
            "solve_time": solve_time, 
            "fitness_features": fitness_features,
            "fitness_score": fitness_score,
            "embedding": code_embedding
        }
    )

    # write code to codedir
    filename = os.path.join(codedir, f"milp_{prefix_id}.py")
    if not os.path.exists(filename):
        open(filename, "w").write(code)

    # save detailed info of this instance
    outdir = os.path.join(details_outdir, f"{prefix_id}_{model_name}")
    os.makedirs(outdir, exist_ok=True)
    # save code_embedding as json
    code_embedding_filename = os.path.join(outdir, "code_embedding.json")
    open(code_embedding_filename, "w").write(json.dumps(code_embedding, indent=4))
    prefixes[-1]["embedding"] = code_embedding_filename

    for keyword in ["milp", "prompt", "output", "system_message", "log"]:
        if keyword in prefixes[-1]:
            filename = os.path.join(outdir, f"milp.py" if keyword == "milp" else f"{keyword}.txt")
            if not os.path.exists(filename):
                open(filename, "w").write(prefixes[-1][keyword])
            # if keyword != "milp":
            prefixes[-1][keyword] = filename

    # save statistics
    if len(prefixes) % 100 == 0 or time.time() - last_cache_time > 60:
        print(f"... Saving {len(prefixes)} milp models ...")
        json.dump(prefixes, open(out_name, "w"), indent=2)
        last_cache_time = time.time()
    return True, prefixes, last_cache_time



def evolve_batch(milp_list, formulation_list, topic_list, 
                 out_name, fitness_criteria=['all'], model_name="gpt-4-1106-preview", embedding_model_name="text-embedding-ada-002", 
                 max_prefix=int(1e6), max_depth=float('inf'), 
                 n_sample_formulation_methods=1, validity_timeout_duration=5, feas_timeout_duration=360, feas_subprocess_timeout_duration=480, 
                 gs_timeout_duration=360, gs_subprocess_timeout_duration=480, 
                 max_select_per_depth=1000, n_cpus=10, gs_n_samples_per_idx=72, gs_range_direction='inc', 
                 gs_n_round=1, gs_final_time_th=5, restart_depth_th=5, fail_th=3, 
                 remove_tmp=True, resume=True, save_satisfy=True,
                 PROMPTS_WEIGHTS_DICT=PROMPTS_WEIGHTS_DICT,
                 seed_wt=0, evolve_wt=0) -> None:

    save_dir = os.path.dirname(out_name)
    codedir = os.path.join(save_dir, "code")
    details_outdir = os.path.join(save_dir, "detailed_info")
    param_search_dir = os.path.join(save_dir, "param_search_dir")
    param_search_tmp_dir = os.path.join(save_dir, "param_search_dir_tmp")
    tmp_dir = os.path.join(save_dir, "tmp_dir")  # temporary directory to check feasibility
    prev_depth_file_name = os.path.join(os.path.dirname(out_name), "prev_depth.json")

    for outdir in [codedir, details_outdir, param_search_dir, param_search_tmp_dir, tmp_dir]:
        os.makedirs(outdir, exist_ok=True)

    ##################################### Init, load seed prefixes #####################################
    prefixes = []
    if os.path.exists(out_name):
        prefixes = json.load(open(out_name, "r"))
    
    if len(prefixes) == 0:
        tasks = []
        for i_milp, milp in enumerate(milp_list):
            filename = os.path.join(codedir, f"milp_{i_milp}.py")
            if not os.path.exists(filename):
                open(filename, "w").write(milp)  

            tasks.append((i_milp, filename, feas_timeout_duration, feas_subprocess_timeout_duration, True, remove_tmp))  # verbose=True
        # parallel check code validity and feasibility
        feasible_results = parallel_fn(check_seed_feasibility, tasks, n_cpus)

        for i_milp, (milp, (feasible, solve_time, log)) in enumerate(zip(milp_list, feasible_results)):
            if feasible:
                prefix_id = len(prefixes)
                if prefix_id != i_milp and os.path.exists(os.path.join(codedir, f"milp_{i_milp}.py")):
                    shutil.move(os.path.join(codedir, f"milp_{i_milp}.py"), 
                                os.path.join(codedir, f"milp_{prefix_id}.py"))

                log_data_list = [log]
                code_embedding = reply_oai_embed(milp, model_name=embedding_model_name)

                if code_embedding is None: 
                    print(f"Fail to get code embedding for code of seed id {len(prefixes)} ... Debug me!")
                    continue

                success, fitness_features, _ = get_fitness_score(code_embedding, log_data_list, [], fitness_criteria=fitness_criteria)
                if not success:
                    print(f"Fail to get fitness score for code of seed id {len(prefixes)} ... Debug me!")
                    continue

                prefixes.append({
                    "milp": milp, 
                    "id": prefix_id, 
                    "source": "algorithm",
                    "root": prefix_id, 
                    "evolve": "Init",
                    "system_message": "Init", 
                    "prompt": "Init",  
                    "output": "Init",
                    "depth": 0, 
                    "log": log,
                    "solve_time": solve_time, 
                    "fitness_features": fitness_features,
                    "fitness_score": 1,   # all init seeds have the same fitness score
                    "embedding": code_embedding
                })


                outdir = os.path.join(details_outdir, f"{prefix_id}_algorithm")
                os.makedirs(outdir, exist_ok=True)
                print(f'Seed {i_milp}:: Write details to {outdir}...')

                # save code to code_dir
                open(os.path.join(codedir, f"milp_{prefix_id}.py"), "w").write(milp)
                # save code_embedding as json
                code_embedding_filename = os.path.join(outdir, "code_embedding.json")
                open(code_embedding_filename, "w").write(json.dumps(code_embedding, indent=4))
                prefixes[-1]["embedding"] = code_embedding_filename
                # store detailed directory
                for keyword in ["milp", "prompt", "output", "system_message", "log"]:
                    if keyword in prefixes[-1]:
                        filename = os.path.join(outdir, f"milp.py" if keyword == "milp" else f"{keyword}.txt")
                        if not os.path.exists(filename):
                            open(filename, "w").write(prefixes[-1][keyword])
                        # if keyword != "milp":
                        prefixes[-1][keyword] = filename
            else:
                print(f'Seed {i_milp}:: Infeasible seed code. Skip.')
            
        json.dump(prefixes, open(out_name, "w"), indent=2)
    ################################# Start evolve #################################
    print('Start evolve! Seed files', len(prefixes), 'out_name', out_name)

    last_cache_time = time.time()
    # The previous depth that we generate the prefixes from
    if not os.path.exists(prev_depth_file_name):
        prev_depth = max(max([prefix["depth"] for prefix in prefixes])-1, 0)  # start from the last depth minus one to move forward
        with open(prev_depth_file_name, 'w') as f:
            json.dump({'prev_depth': prev_depth}, f, indent=2)
    else:
        prev_depth = json.load(open(prev_depth_file_name, 'r'))['prev_depth']

    cur_depth = prev_depth + 1    # current depth
    fail_depth = 0  # how many times we fail to generate enough prefixes at the current depth

    while len(prefixes) < max_prefix and cur_depth < max_depth:
        print(f'----------------------------------------- start depth {cur_depth} -----------------------------------------')
        # evolve all prefixes from the same depth
        prefixes_depth = get_prefixes_depth(prefixes, prev_depth, fail_depth, fail_th=fail_th, restart_depth_th=restart_depth_th)
        depth_data_dir = os.path.join(tmp_dir, f"prompts_{cur_depth}_tmp")
        depth_data_path = os.path.join(depth_data_dir, 'depth_data.jsonl')
        prompt_list, new_prefix_list = gen_prompts_depth(prefixes, cur_depth, prefixes_depth, 
                                                                depth_data_path, formulation_list, topic_list, 
                                                                n_sample_formulation_methods=n_sample_formulation_methods,
                                                                max_select_per_depth=max_select_per_depth,
                                                                resume=resume, solve_time_high=gs_timeout_duration-30,
                                                                PROMPTS_WEIGHTS_DICT=PROMPTS_WEIGHTS_DICT,
                                                                seed_wt=seed_wt, evolve_wt=evolve_wt)
                                    
        print(f'----------------------------------------- Generated {len(prompt_list)} prompts at depth {cur_depth} --------------------------------------')

        ################# query openai to get the answers #################
        save_data_list = []
        for i, (system_message, prompt, evolve, id_from) in enumerate(new_prefix_list):
            ans = do_oai_reply(system_message, prompt, model_name=model_name)
            save_data_list.append((i, system_message, prompt, evolve, id_from, ans))

        ################### process the answers from gpt queries ###################
        print(f"---------------------- answers length {len(save_data_list)} prompt length {len(prompt_list)} ----------------------")

        tasks = []
        save_data_list_task = []
        existing_codes = set([open(x["milp"]).read() for x in prefixes])
        for i, system_message, prompt, evolve, id_from, ans in save_data_list:
            tmp_idx = len(prefixes) + i
            code = extract_code_block(ans)

            if code is None or len(code) < 10 or code in existing_codes:
                continue # the generated code is too short or already exsits

            tasks.append((code, tmp_dir, tmp_idx, validity_timeout_duration, feas_timeout_duration, feas_subprocess_timeout_duration, remove_tmp))
            save_data_list_task.append((i, system_message, prompt, evolve, id_from, ans))
        
        del existing_codes  # free memory

        # parallel check code validity and feasibility
        feasible_results = parallel_fn(check_evolve_feasbility, tasks, n_cpus)
        print(f"[Start param search] ------------- # feasible results for parameter grid search {len([x for x in feasible_results if x[0]])} out of {len(feasible_results)} -------------")
        ################### for each code, parallel do parameter search to see if the code can be improved
        for ((feasible, code, _, log), (i, system_message, prompt, evolve, id_from, ans)) in zip(feasible_results, save_data_list_task):
            if not feasible: continue
            ############################ parameter search for the generated code ################################
            param_search_dir_i = os.path.join(param_search_dir, f"{len(prefixes)}_{model_name}")
            save_tmp_file_path_i = os.path.join(param_search_tmp_dir, f"{len(prefixes)}_{model_name}")
            save_satisfy_file_path_i = os.path.join(param_search_tmp_dir, f"{len(prefixes)}_{model_name}_satisfy")
            for param_dir_i, make_dir in [(param_search_dir_i, True), (save_tmp_file_path_i, True), (save_satisfy_file_path_i, save_satisfy)]:
                if make_dir and param_dir_i and not os.path.exists(param_dir_i):
                    os.makedirs(param_dir_i, exist_ok=True)

            success, code, log = param_search_i(len(prefixes), param_search_dir_i, code, log, gs_n_samples_per_idx, gs_timeout_duration,
                                                gs_subprocess_timeout_duration, gs_range_direction, gs_n_round, gs_final_time_th, n_cpus, 
                                                remove_tmp, save_tmp_file_path_i, save_satisfy, save_satisfy_file_path_i)

            if success:
                code_embedding = reply_oai_embed(code, model_name=embedding_model_name)
                if code_embedding is None:
                    print(f"Fail to get code embedding for code of id {len(prefixes)} ... Debug me!")
                ################################ save data ################################

                success, prefixes, last_cache_time = save_data_i(code, log, system_message, prompt, ans, code_embedding,
                                                                    prefixes, evolve, id_from, cur_depth, param_search_dir_i, 
                                                                    model_name, fitness_criteria, codedir, details_outdir, out_name, last_cache_time)
            if not success:
                for param_dir in [param_search_dir_i, save_tmp_file_path_i, save_satisfy_file_path_i]:
                    if param_dir and os.path.exists(param_dir):
                        shutil.rmtree(param_dir)

            if remove_tmp and os.path.exists(save_tmp_file_path_i):
                shutil.rmtree(save_tmp_file_path_i)

        ################### current level done: declare success and move on to the next level if at least restart_depth_th are generated
        num_milps_cur_depth = len([prefix for prefix in prefixes if prefix["depth"] == cur_depth])
        if num_milps_cur_depth >= restart_depth_th:
            prev_depth = cur_depth
            cur_depth += 1
            ################### remove previous level prompt i_depth data to save memory, as we are moving to cur_depth+1
            prev_depth_data_dir = os.path.join(tmp_dir, f"{prev_depth}_tmp")
            if os.path.exists(prev_depth_data_dir):
                shutil.rmtree(prev_depth_data_dir)
            fail_depth = 0
            print(f'Successfully finish {prev_depth} with {num_milps_cur_depth} milps generated. Move to the next depth {cur_depth} ...')
            # replace out_name file name with max_depth.json save max depth to json
            with open(prev_depth_file_name, 'w') as f:
                json.dump({'prev_depth': prev_depth}, f, indent=2)
        else:
            # remove the current depth data to regenerate prompt for the current depth
            if os.path.exists(depth_data_dir):
                shutil.rmtree(depth_data_dir)  
            fail_depth += 1     
            print(f'Remove current depth data to regenerate prompt for the current depth {cur_depth}. Failing {fail_depth} times ...')

