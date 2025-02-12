import ast
import os
import pdb
import random
import re
import subprocess
import threading
import time
from datetime import datetime
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import tqdm
from dotenv import load_dotenv
from joblib import Parallel, delayed
from openai import OpenAI


load_dotenv('../.env')
api_key = os.getenv('OPENAI_API_KEY', "")

client = embed_client = OpenAI(
    api_key=api_key,
)

# parameter search parameters
TIMEOUT_DURATION = 300  # SCIP solve time limit
SUBPROCESS_TIMEOUT_DURATION = 600  # SCIP solve time limit
GAP_TH = 0  # Tolerated gap at time limit



def reply_oai(prompt, system_message="", model_name="gpt-4o"):
    # if prompt is str:
    if isinstance(prompt, str):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    else:
        system_message = [data['content'] for data in prompt if data['role'] == 'system'][0]
        user_message = [data['content'] for data in prompt if data['role'] == 'user'][0]

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
   
    return response.choices[0].message.content


def do_oai_reply(system_message="", prompt="", model_name="gpt-4o"):
    responses_i = []
    retry_count = 0
    while len(responses_i) < 1:
        try:
            responses_i = reply_oai(
                prompt=[
                    {"content": system_message, "role": "system"},
                    {"content": prompt, "role": "user"},
                ],
                model_name=model_name
            )
            if isinstance(responses_i, str):
                responses_i = [responses_i]

        except Exception as e:
            e = str(e)
            if 'html' in e and "<div id='message'>" in e:
                e = e.split("<div id='message'>")[1].split("</div>")[0]
            print('OAI reply exception::', e[-100:])
            time.sleep(random.uniform(10, 30))
            retry_count += 1
            if retry_count > 20:
                print("Maximum retry limit reached. Exiting.")
                return ""
            else:
                continue
    if isinstance(responses_i, str):
        responses_i = [responses_i]
        
    return responses_i[0] if len(responses_i) > 0 else ""


def reply_oai_embed(text, model_name="text-embedding-ada-002"):
    if isinstance(text, str):
        text = [text]

    responses_i = []
    retry_count = 0
    while len(responses_i) < 1:
        try:
            responses_i = embed_client.embeddings.create(model=model_name, input=text)            
            if not isinstance(responses_i, list) and not isinstance(responses_i, tuple):
                responses_i = [responses_i]
        except Exception as e:
            print('OAI embedding exception::', e)
            time.sleep(random.uniform(5, 30))
            retry_count += 1
            if retry_count > 20:
                print('Failed to generate embeddings! Return None')
                return None
            else:
                continue
    if not isinstance(responses_i, list) and not isinstance(responses_i, tuple):
        responses_i = [responses_i]
    
    return responses_i[0].data[0].embedding if len(responses_i) > 0 else None
    

#### Multiprocess helper function
def multiprocess(func, tasks, n_cpus=None):
    if n_cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    with Pool(n_cpus or os.cpu_count()) as pool:
        return list(pool.imap(func, tasks))


def multithread(func, tasks, n_cpus=None, show_bar=True):
    bar = lambda x: tqdm(x, total=len(tasks)) if show_bar else x
    if n_cpus == 1 or len(tasks) == 1:
        return [func(t) for t in bar(tasks)]
    with ThreadPool(n_cpus or os.cpu_count()) as pool:
        return list(bar(pool.imap(func, tasks)))
    
def parallel_fn(func, tasks, n_cpus=None):
    if len(tasks) == 0:
        print('No tasks to run')
        return []
    
    if n_cpus == 0 and len(tasks) > 0:
        print(f'n_cpus is 0 but len(tasks) = {len(tasks)}, set n_cpus to 1')
        n_cpus = 1  
    

    if n_cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    
    n_cpus = os.cpu_count() if n_cpus is None else min(n_cpus, os.cpu_count()) 

    return Parallel(n_jobs=min(n_cpus, len(tasks)))(delayed(func)(t) for t in tasks)


#### Helper function to parse the MILP code file
def add_time_limit(file_path, time_limit, tmp_file_path=None):
    if tmp_file_path is None:
        tmp_file_path = file_path if 'tmp' in file_path else f"{os.path.splitext(file_path)[0]}_tmp.py"

    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if "def solve" in line:
            add_time_limit_next = True
        if "model.optimize()" in line and add_time_limit_next:
            leading_whitespace = line[:len(line) - len(line.lstrip())]
            time_limit_line = f'{leading_whitespace}model.setParam("limits/time", {time_limit})\n'
            new_lines.append(time_limit_line)
            add_time_limit_next = False
        new_lines.append(line)

    new_lines = remove_duplicate(new_lines)

    with open(tmp_file_path, 'w') as file:
        file.writelines(new_lines)

    return tmp_file_path


### call function helper: call another python file to check if there are bugs
def check_valid_file(script_path, timeout_duration=5):
    try:
        result = subprocess.run(
            ['python', script_path], 
            capture_output=True, 
            text=True, 
            timeout=timeout_duration
        )
        
        # Check if there was any error
        if result.returncode != 0:
            return False  # f"Script has bugs. Error: {result.stderr.strip()}"
        else:
            return True  # "Script ran successfully. No bugs detected."
    
    except subprocess.TimeoutExpired:
        return True  # "Script timed out. It might be stuck or have performance issues."
    except Exception as e:
        return False  # f"An error occurred while running the script: {str(e)}"


def get_status_time_and_gap(text, last_n=15):
    try:
        lines = text.split('\n')[-last_n:]
    except:
        print('Get status error', text)
        pdb.set_trace()
        exit()

    solve_status = None
    solve_time = None
    gap = None
    for line in lines:
        if "Solve Status:" in line:
            solve_status_match = re.search(r"Solve Status:\s*(.*)", line)
            if solve_status_match:
                solve_status = solve_status_match.group(1).strip()
        if "Solve Time:" in line:
            solve_time_match = re.search(r"Solve Time:\s*([\d.]+)\s*seconds", line)
            if solve_time_match:
                solve_time = float(solve_time_match.group(1))
        if "Gap" in line:
            # Gap                : 0.00 %
            # Gap                : infinite
            gap_match = re.search(r"Gap\s*:\s*([\d.]+\s*)%", line)
            if gap_match:
                gap = float(gap_match.group(1))

    # If solve status is None, check for SCIP Status patterns
    if solve_status is None:
        for line in lines:
            if line.startswith("SCIP Status"):
                if "problem is solved [optimal solution found]" in line:
                    solve_status = "optimal"
                elif "problem is solved [infeasible]" in line:
                    solve_status = "infeasible"
                elif "solving was interrupted [time limit reached]" in line:
                    solve_status = "timelimit"
                break
    
    # If solve time is None, check for Solving Time patterns
    if solve_time is None:
        for line in lines:
            if line.startswith("Solving Time (sec)"):
                solve_time_match = re.search(r"Solving Time \(sec\)\s*:\s*([\d.]+)", line)
                if solve_time_match:
                    solve_time = float(solve_time_match.group(1))
                break
    
    # If any of the values are still None, set them all to None (unsuccessful run)
    if solve_time is None or solve_status is None or gap is None:
        solve_status = solve_time = gap = None

    return solve_status, solve_time, gap


def run_script_from_tmp(tmp_script_path, subprocess_timeout_duration, remove_tmp=True):  # , logfile=None
    result = subprocess.run(
        ['python', tmp_script_path],
        capture_output=True,
        text=True,
        timeout=subprocess_timeout_duration
    )
    # remove tmp file
    if remove_tmp and os.path.exists(tmp_script_path):
        os.remove(tmp_script_path)
    return result


### If can run more than 20 seconds: assume feasible
def check_feasibility(script_path, timeout_duration=TIMEOUT_DURATION, 
                      subprocess_timeout_duration=SUBPROCESS_TIMEOUT_DURATION, 
                      last_n=10, verbose=True, gap_th=GAP_TH, remove_tmp=True):
    tmp_script_path = add_time_limit(script_path, timeout_duration)
    start_time = time.time()
    
    result = None
    try:
        result = run_script_from_tmp(tmp_script_path, subprocess_timeout_duration, remove_tmp=remove_tmp) # , logfile=tmp_file_log

        # If script has any error
        if result.returncode != 0:
            if verbose:
                print(f'Return code {result.returncode}!', tmp_script_path, f'time last {time.time() - start_time:.2f}s') # , 'output', result.stdout
            # return False, None, result.stdout  # Indicates the script has bugs or errors

        # Parse the last_n lines of the output for solve status and solve time
        solve_status, solve_time, gap = get_status_time_and_gap(result.stdout, last_n=last_n)

        # Check if solve status indicates success: feasible or optimal
        if solve_status is None:
            print('Solve status is None! Solve time', solve_time, 'tmp_script_path', tmp_script_path)
            return False, None, result.stdout

        if solve_status.lower() == 'timelimit':
            print(f'{script_path} time limit ({subprocess_timeout_duration} seconds) exceeded. Please manually check the log file.')

        if solve_status.lower() in ['feasible', 'optimal']:
            return True, solve_time, result.stdout
        elif solve_status.lower() == 'timelimit':
            return gap is not None and gap <= gap_th, solve_time, result.stdout
        else:
            return False, solve_time, result.stdout

    except subprocess.TimeoutExpired as timeErr:
        if verbose:
            print(f'{script_path} Timeout! time last {time.time() - start_time:.2f}s')

        if remove_tmp and os.path.exists(tmp_script_path):
            os.remove(tmp_script_path)

        return False, None, ''

    except Exception as e:
        if verbose:
            print(f'{script_path} Other exception: {e}')

        if remove_tmp and os.path.exists(tmp_script_path):
            os.remove(tmp_script_path)

        return False, None, ''



def run_and_save_log(script_path, log_path, timeout_duration=TIMEOUT_DURATION, subprocess_timeout_duration=SUBPROCESS_TIMEOUT_DURATION, 
                     verbose=True, remove_tmp=True):
    # the file should be feasible to run
    tmp_script_path = add_time_limit(script_path, timeout_duration)
    try:
        result = run_script_from_tmp(tmp_script_path, subprocess_timeout_duration, remove_tmp=remove_tmp)

        with open(log_path, 'w') as log_file:
            log_file.write(result.stdout)
        return True

    except subprocess.TimeoutExpired:
        # remove tmp file
        if verbose:
            print(f'{script_path} Timeout!')

        if remove_tmp and os.path.exists(tmp_script_path):
            os.remove(tmp_script_path)
        return False

    except Exception as e:
        if verbose:
            print(f'{script_path} Error: {e}')

        if remove_tmp and os.path.exists(tmp_script_path):
            os.remove(tmp_script_path)

        return False


# modify the seed of the current file to see if the code will be feasible under another distribution
def modify_seed(script_path, seed_number, new_seed):
    with open(script_path, 'r') as file:
        file_content = file.read()
    
    # Regular expression to find the seed line
    seed_pattern = re.compile(r'^(\s*)seed\s*=\s*\d+', re.MULTILINE)
    
    def replace_seed(match):
        leading_spaces = match.group(1)
        return f'{leading_spaces}seed = {new_seed}'

    modified_content = seed_pattern.sub(replace_seed, file_content)
    
    base_name, ext = os.path.splitext(script_path)
    new_file_name = f'{base_name}_{seed_number}{ext}'
    with open(new_file_name, 'w') as new_file:
        new_file.write(modified_content)
    
    return new_file_name


### call function helper: call code directly
def check_valid_code(code_str, timeout_duration=3):
    print('code_str', code_str)
    def target():
        try:
            exec(code_str)
        except Exception as e:
            result.append(f"Code has bugs. Error: {str(e)}")

    result = []
    thread = threading.Thread(target=target)

    # Start the thread and enforce a 5-second timeout
    thread.start()
    thread.join(timeout=timeout_duration)

    if thread.is_alive():
        thread._stop()  # Forcefully stop the thread (use with caution)

    return len(result) == 0


####### code blocks helper functions
# PROMPT2 get separate code blocks
given_data_marker = "### given instance data code ends here"
new_data_marker = "### new instance data code ends here"
given_constraints_marker = "### given constraints and variables and objective code ends here"
new_constraints_marker = "### new constraints and variables and objective code ends here"
given_parameters_marker = "### given parameter code ends here"
new_parameters_marker = "### new parameter code ends here"


# PROMPT1 get the complete code
def remove_duplicate(filtered_lines):
    # Find the last occurrence of `model.optimize()`
    last_optimize_index = -1
    for i, line in enumerate(filtered_lines):
        if 'model.optimize()' in line:
            last_optimize_index = i
    
    # Keep only the last `model.optimize()`
    if last_optimize_index != -1:
        filtered_lines = [line for i, line in enumerate(filtered_lines) if 'model.optimize()' not in line or i == last_optimize_index]
    
    return filtered_lines


def extract_code_block(text):
    if not text or not isinstance(text, str):
        return None
    # Regular expression to capture the content between ```python and ```
    code_block_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
    
    match = code_block_pattern.search(text)
    if match:
        markers = [
            'given_data_marker',
            'new_data_marker',
            'given_constraints_marker',
            'new_constraints_marker',
            'given_parameters_marker',
            'new_parameters_marker'
        ]
        code_block = match.group(1).strip()
        lines = code_block.split('\n')
        filtered_lines = [line for line in lines if all(marker not in line for marker in markers)]
        filtered_lines = remove_duplicate(filtered_lines)

        code_block = "\n".join(filtered_lines)
        return code_block
    else:
        return None


def extract_code_blocks(text):
    # Define regex patterns for extracting code blocks
    constraint_pattern = r"### New constraint code.*?```python(.*?)```.*?### New data code"
    data_pattern = r"### New data code.*?```python(.*?)```.*?### New parameter code"
    parameter_pattern = r"### New parameter code.*?```python(.*?)```"

    # Find code blocks using regex search
    constraint_match = re.search(constraint_pattern, text, re.DOTALL)
    data_match = re.search(data_pattern, text, re.DOTALL)
    parameter_match = re.search(parameter_pattern, text, re.DOTALL)
    
    # Extract and clean code blocks
    new_constraint_code = constraint_match.group(1) if constraint_match else ''
    new_data_code = data_match.group(1) if data_match else ''
    new_parameter_code = parameter_match.group(1) if parameter_match else ''
    
    return new_constraint_code, new_data_code, new_parameter_code


def insert_comments(code):
    # Define the pairs of lines to be inserted with a new line between them
    combined_lines_to_insert = [
        (f"{given_data_marker}", new_data_marker),
        (f"{given_constraints_marker}", new_constraints_marker),
        (f"{given_parameters_marker}", new_parameters_marker)
    ]

    # Define the insertion points for each pair of lines
    insertion_points = [
        ("return res", combined_lines_to_insert[0]),
        ("model.setObjective(", combined_lines_to_insert[1])
    ]

    def get_indentation(line):
        return len(line) - len(line.lstrip())

    def insert_with_indentation(lines, idx, line_to_insert):
        previous_line = lines[idx]
        indentation = len(previous_line) - len(previous_line.lstrip())
        indented_line = ' ' * indentation + line_to_insert
        lines.insert(idx + 1, indented_line)
        
    # Split the original code into lines for easier processing
    org_lines = code.split('\n')
    new_lines = []
    # Iterate through lines and manage insertions
    for idx, line in enumerate(org_lines):
        # Check if we need to insert lines before this line
        for insertion_point, insert_lines in insertion_points:
            if insertion_point in line:
                # Insert new lines before the current line
                indentation_level = get_indentation(line)
                for new_line in insert_lines:
                    new_lines.append(' ' * indentation_level + new_line)

        # Add the current line to new_lines
        new_lines.append(line)

    latest_param_index = -1
    # Iterate through lines to find the latest parameter assignment
    idx = 0
    param_dict_parsed = False
    while idx < len(new_lines):
        line = new_lines[idx]

        # Detect parameter assignments
        if not param_dict_parsed:
            if line.strip().startswith('parameters = {'):
                latest_param_index = idx
                idx += 1
                # Move to the end of the block
                while not new_lines[idx].strip().endswith('}'):
                    idx += 1
                param_dict_parsed = True
                latest_param_index = idx
                # print('End of param parse', idx, new_lines[idx])    
        elif line.strip().startswith('parameters['):
            latest_param_index = idx

        idx += 1

    # Insert comments after the latest parameter assignment
    if latest_param_index != -1:
        insert_with_indentation(new_lines, latest_param_index, combined_lines_to_insert[2][0])
        insert_with_indentation(new_lines, latest_param_index + 1, combined_lines_to_insert[2][1])

    return '\n'.join(new_lines)


#### util functions to change the seed in the file (and change from maximize to minimize of negative objective)
def change_seed_in_file(file_path, new_seed, tmp_file_path=None, mps_file=None):
    with open(file_path, 'r') as original_file:
        lines = original_file.readlines()

    seed_pattern = re.compile(r'(seed\s*=\s*)\d+')
    objective_pattern_start = re.compile(r'(\s*)model\.setObjective\(\s*')
    objective_pattern_maximize = re.compile(r'\s*["\']maximize["\']\s*')
    objective_pattern_end = re.compile(r'\s*\)\s*')

    if tmp_file_path is None:
        tmp_file_path = f"{os.path.splitext(file_path)[0]}_seed{new_seed}.py"


    if '"maximize"' in ''.join(lines) or '\'maximize\'' in ''.join(lines):
        objective_found = False
        buffer = []
        inside_objective = False
        maximize_found = False
        seed_found = False

        with open(tmp_file_path, 'w') as file:
            for line in lines:
                # Update the seed
                if not seed_found and re.search(seed_pattern, line):
                    seed_found = True
                    line = re.sub(seed_pattern, r'\g<1>' + str(new_seed), line)
                    file.write(line)
                    continue

                # Check for the start of the objective
                if not objective_found and re.search(objective_pattern_start, line):
                    inside_objective = True

                if inside_objective:
                    buffer.append(line)

                    # Check for maximize within the objective
                    if re.search(objective_pattern_maximize, line):
                        line = re.sub(objective_pattern_maximize, lambda match: match.group(0).replace('\'maximize\'', '"minimize"').replace('"maximize"', '"minimize"'), line)
                        buffer[-1] = line
                        maximize_found = True

                    # Check for the end of the objective only if maximize was found
                    if maximize_found and re.search(objective_pattern_end, line):
                        objective_found, inside_objective, maximize_found = True, False, False
                        full_objective = ''.join(buffer)
                        new_objective_expr = full_objective.replace('model.setObjective(', 'model.setObjective(-(')
                        new_objective_expr = re.sub(r'(,\s*\n*\s*)"minimize"', r')\1"minimize"', new_objective_expr)
                        new_objective_expr = re.sub(r"(,\s*\n*\s*)'minimize'", r")\1'minimize'", new_objective_expr)
                        file.write(new_objective_expr)
                        buffer = []
                    continue

                # Write to the temporary file
                if mps_file is not None and 'model.optimize()' in line:
                    indentation = len(line) - len(line.lstrip())
                    problem_line = ' ' * indentation + f"model.writeProblem('{mps_file}')\n"
                    file.write(problem_line)
                    file.write(line)
                    continue

                file.write(line)
                    
        # Save to debug_mps.py if the objective line is not found
        if not objective_found:
            print('Maximize: Objective line not found in', file_path)
            with open(tmp_file_path, 'w') as file:
                file.writelines(lines)

            debug_file_path = "debug_mps.py"
            with open(debug_file_path, 'w') as debug_file:
                debug_file.writelines(lines)
    else:
        seed_found = False
        with open(tmp_file_path, 'w') as file:
            for line in lines:
                # Update the seed
                if not seed_found and re.search(seed_pattern, line):
                    seed_found = True
                    line = re.sub(seed_pattern, r'\g<1>' + str(new_seed), line)
                    file.write(line)
                    continue

                # Write to the temporary file
                if mps_file is not None and 'model.optimize()' in line:
                    indentation = len(line) - len(line.lstrip())
                    problem_line = ' ' * indentation + f"model.writeProblem('{mps_file}')\n"
                    file.write(problem_line)
                    file.write(line)
                    continue

                file.write(line)

    # Verify file is not empty or just space
    with open(tmp_file_path, 'r') as file:
        if not any(line.strip() for line in file):
            print('Empty file', tmp_file_path)

            with open(file_path, 'r') as new_file:
                if not any(line.strip() for line in new_file):
                    print('Empty original file', file_path)

    return tmp_file_path

    

def change_seed_in_file_simple(file_path, new_seed, tmp_file_path=None, mps_file=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    seed_pattern = r'(seed\s*=\s*)\d+'

    if tmp_file_path is None:
        tmp_file_path = f"{os.path.splitext(file_path)[0]}_seed{new_seed}.py"

    with open(tmp_file_path, 'w') as file:
        for line in lines:
            if re.search(seed_pattern, line):
                line = re.sub(seed_pattern, r'\g<1>' + str(new_seed), line)
            
            if mps_file is not None and 'model.optimize()' in line:
                indentation = len(line) - len(line.lstrip())
                file.write(' ' * indentation + f"model.writeProblem('{mps_file}')\n")
                file.write(line)
            else:
                file.write(line)

    # verify file is not empty or just space
    with open(tmp_file_path, 'r') as file:
        if not any(line.strip() for line in file):
            print('Empty file', tmp_file_path)

            with open(file_path, 'r') as new_file:
                if not any(line.strip() for line in new_file):
                    print('Empty original file', file_path)

    return tmp_file_path


def change_seed_and_check_feasibility(file_path, new_seed, timeout_duration=TIMEOUT_DURATION, 
                                      subprocess_timeout_duration=SUBPROCESS_TIMEOUT_DURATION,
                                      mps_file=None, gap_th=GAP_TH, remove_tmp=True, tmp_file_path=None):
    tmp_file_path = change_seed_in_file(file_path, new_seed, mps_file=mps_file, tmp_file_path=tmp_file_path)
    feasible, solve_time, output = check_feasibility(tmp_file_path, timeout_duration=timeout_duration, 
                                                     subprocess_timeout_duration=subprocess_timeout_duration,
                                                     gap_th=gap_th)
    # remove tmp file
    if remove_tmp and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    return feasible, solve_time, output


def change_seed_and_check_valid_file(file_path, new_seed, timeout_duration=10,
                                     mps_file=None, remove_tmp=True, tmp_file_path=None):
    tmp_file_path = change_seed_in_file(file_path, new_seed, mps_file=mps_file, tmp_file_path=tmp_file_path)
    valid = check_valid_file(tmp_file_path, timeout_duration=timeout_duration)
    # remove tmp file
    if remove_tmp and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    return valid


#### util function to change the parameter values in the file
def parse_parameters_from_content(file_content, default_value=0):
    def get_parameters_str(file_content): 
        extracting = False           
        dict_content = []
        start_pattern = re.compile(r'parameters\s*=\s*{')
        empty_pattern = re.compile(r'parameters\s*=\s*{\s*}')

        for line in file_content.split('\n'):
            line = re.sub(r'#.*', '', line).strip()

            if empty_pattern.match(line):
                return []
        
            if start_pattern.search(line):
                extracting = True
                split_line = start_pattern.split(line)
                if len(split_line) > 1:
                    dict_content.append(split_line[1])
                continue
            
            if extracting:
                if line.strip().startswith('}'):
                    break
                dict_content.append(line)
        return dict_content


    def parse_value_node(node):
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant):  # Python 3.8+ handling
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.List):
            return [parse_value_node(element) for element in node.elts]
        elif isinstance(node, ast.Dict):
            return {parse_value_node(k): parse_value_node(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Tuple):
            return tuple(parse_value_node(element) for element in node.elts)
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, (ast.Call, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            return ast.unparse(node) if hasattr(ast, 'unparse') else default_value
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else default_value

    
    dict_content = get_parameters_str(file_content)

    parameters = {}
    if dict_content:
        # Create one complete string from the list and properly format it to close the dictionary
        dict_str = '{' + ''.join(dict_content) + '}'
        dict_str = dict_str.replace('\n', ' ')            
        try:
            dict_node = ast.parse(dict_str, mode='eval').body
            if isinstance(dict_node, ast.Dict):
                for key_node, value_node in zip(dict_node.keys, dict_node.values):
                    # Extract key
                    if isinstance(key_node, ast.Str):
                        key = key_node.s
                    elif isinstance(key_node, ast.Constant):  # Python 3.8+ handling
                        key = key_node.value
                    else:
                        raise ValueError("Dictionary keys must be strings or constants.")
                    
                    value = parse_value_node(value_node)
                    parameters[key] = value
        except:
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print('Error parsing Dictionary string:', dict_str)
            os.makedirs('errors', exist_ok=True)
            current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            open(f'errors/error-{current_time_str}.txt', 'w').write(file_content)
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            parameters = {}
        
    return parameters


def parse_parameters(file_path):
    file_content = open(file_path, "r").read()

    return parse_parameters_from_content(file_content)


def get_parameters_str(file_path):
    file_content = open(file_path, "r").read()
    
    # Regex pattern to find the parameters dictionary
    pattern = re.compile(r'\s*parameters\s*=\s*\{.*?\}', re.DOTALL)
    match = pattern.search(file_content)
    return match.group(0) if match else ""


def edit_parameters_from_content(file_content, new_parameters, tmp_file_path=None):
    lines = file_content.splitlines()
    new_lines = []
    inside_params = False
    indentation = ''
    
    for line in lines:
        if inside_params:
            if line.strip().endswith('}'):
                inside_params = False
                continue
            else:
                continue

        if re.match(r'\s*parameters\s*=\s*\{', line):
            inside_params = True
            indentation = len(line) - len(line.lstrip())
            # Create a string representation of the new parameters dictionary
            new_lines.append(line[:indentation] + "parameters = {")
            for key, value in new_parameters.items():
                new_lines.append(' ' * (indentation + 4) + f"'{key}': {repr(value)},")
            new_lines.append(' ' * indentation + '}')
        else:
            new_lines.append(line)

    new_content = '\n'.join(new_lines)
    return new_content

    if tmp_file_path is None:
        tmp_file_path = f"{os.path.splitext(file_path)[0]}_tmp.py"
    with open(tmp_file_path, 'w') as tmp_file:
        tmp_file.write(new_content)


def edit_parameters(file_path, new_parameters):
    file_content = open(file_path, "r").read()

    return edit_parameters_from_content(file_content, new_parameters)
    

#### scip statistics parsing
def extract_scip_statistics(log_file):
    stats = {}
    
    with open(log_file, 'r') as file:
        log_data = file.read()
    
    # Extract key statistics using regex
    stats_patterns = {
        # Problem statistics
        "SCIP Status": (r"SCIP Status\s*:\s*(.+)", str),
        "Original Variables": (r"Original Problem\s+Variables\s*:\s*(\d+)", int),
        "Original Constraints": (r"Original Problem\s+Constraints\s*:\s*(\d+)", int),
        "Presolved Variables": (r"Presolved Problem\s+Variables\s*:\s*(\d+)", int),
        "Presolved Constraints": (r"Presolved Problem\s+Constraints\s*:\s*(\d+)", int),
        
        # Solve statistics
        "Total Time": (r"Total Time\s*:\s*([\d.]+)", float),
        "Solving Time": (r"solving\s*:\s*([\d.]+)", float),
        "Presolving Time": (r"presolving\s*:\s*([\d.]+)", float),
        "Reading Time": (r"reading\s*:\s*([\d.]+)", float),
        "Copying Time": (r"copying\s*:\s*([\d.]+)", float),
        
        # Presolve statistics
        "Presolvers Total Time": (r"Presolvers\s+:\s+TotalTime\s*([\d.]+)", float),
        "Fixed Variables": (r"FixedVars\s*:\s*(\d+)", int),
        "Aggregated Variables": (r"AggrVars\s*:\s*(\d+)", int),
        "Deleted Constraints": (r"DelCons\s*:\s*(\d+)", int),
        
        # LP statistics
        "Dual LP Iterations": (r"dual LP\s*:\s*[\d.]+\s+[\d]+\s+([\d]+)", int),
        "LP Iterations at Root": (r"root node\s+:\s+Final Root Iters\s*:\s*(\d+)", int),
        
        # Branching statistics
        "Branch-and-Bound Nodes": (r"B&B Tree\s+:.*nodes\s*:\s*(\d+)", int),
        "Branch-and-Bound Max Depth": (r"B&B Tree\s+:.*max depth\s*:\s*(\d+)", int),
        "Backtracks": (r"backtracks\s*:\s*(\d+)", int),
        
        # Constraints
        "Constraints Separate Time": (r"Constraint Timings\s+:.*Separate\s+([\d.]+)", float),
        "Constraints Propagate Time": (r"Constraint Timings\s+:.*Propagate\s+([\d.]+)", float),
        
        # Other statistics
        "Solutions found": (r"Solutions\s+found\s*:\s*(\d+)", int),
        "Gap": (r"Gap\s*:\s*([\d.]+)\s*%", float),
        "First LP value": (r"First LP value\s*:\s*([\d.+-e]+)", float),
        "First LP Iters": (r"First LP Iters\s*:\s*(\d+)", int),
        "Final Dual Bound": (r"Final Dual Bound\s*:\s*([\d.+-e]+)", float),
        "Root LP Estimate": (r"Root LP Estimate\s*:\s*([\d.+-e]+)", float),
        "Primal Bound": (r"Primal Bound\s*:\s*([\d.+-e]+)", float),
        "Infeasible leaves": (r"infeas\. leaves\s*:\s*(\d+)", int),
        "Objective leaves": (r"objective leaves\s*:\s*(\d+)", int),
        "Average Node Depth": (r"avg switch length\s*:\s*([\d.]+)", float)
    }

    for stat, (pattern, cast_type) in stats_patterns.items():
        match = re.search(pattern, log_data)
        if match:
            try:
                stats[stat] = cast_type(match.group(1))
            except:
                print(stat)
    
    # Cutting planes detailed multi-line pattern matching
    cutting_planes_to_track = [
        "cut pool", "aggregation", "cgmip", "clique", "closecuts",
        "cmir", "convexproj", "disjunctive", "eccuts", "flowcover",
        "gauge", "gomory", "impliedbounds", "intobj", "mcf",
        "oddcycle", "rapidlearning", "strongcg", "zerohalf"
    ]

    # Construct regex dynamically from the list
    cutting_planes_regex = '|'.join(map(re.escape, cutting_planes_to_track))
    pattern = rf"({cutting_planes_regex})\s+:\s+[\d.]+\s*[\d.]*\s*\d*\s*\d*\s*\d*\s*\d*\s*(\d+)"
    
    # Find all relevant cutting plane applied counts
    cutting_planes_stats = re.findall(pattern, log_data, re.MULTILINE)
    for plane, applied in cutting_planes_stats:
        stats[f"{plane} Cuts Applied"] = int(applied)
        
    return stats


def parse_solving_process_from_content(log_data, detailed_logs=False, verbose=False):
    def convert_value(value):
        value = value.strip()
        if value[-1].lower() == 'k':
            return float(value[:-1]) * 1e3
        elif value[-1].lower() == 'm':
            return float(value[:-1]) * 1e6
        return float(value)

    stats = {
        "Total Solving Time": None,
        "Total Nodes": None,
        "First Presolve Stats": None,
        "Total Presolving Time": None,
        "Other Presolve Stats": [],
        "Primal Bound": None,
        "Dual Bound": None,
        "Gap": None,
        "Total Variables": None,
        "Binary Variables": None,
        "Integer Variables": None,
        "Implicit Variables": None,
        "Continuous Variables": None,
        "Total Constraints": None,
        "Constraint Types": {},
        "Number of BnB Logs": None,
        "Number of BnB Nodes": None,
        "First LP val": None,
        "Integrality Gap": None,  # approximate
        "Root Time": None,  # approximate
    }
    if detailed_logs:
        stats["BnB Logs"] = []

    # Extract presolve stats
    presolve_pattern = r"presolving:\n(.*?Presolving Time: [\d.]+)"
    presolve_matches = re.findall(presolve_pattern, log_data, re.DOTALL)

    presolve_stats_pattern = (
        r"presolved problem has (\d+) variables \((\d+) bin, (\d+) int, (\d+) impl, (\d+) cont\) and (\d+) constraints\n"
        r"[\s\S]*?Presolving Time: ([\d.]+)"
    )

    if presolve_matches:
        first_presolve_stats = re.search(presolve_stats_pattern, presolve_matches[0])
        if first_presolve_stats:
            stats["First Presolve Stats"] = {
                "Total Variables": int(first_presolve_stats.group(1)),
                "Binary Variables": int(first_presolve_stats.group(2)),
                "Integer Variables": int(first_presolve_stats.group(3)),
                "Implicit Variables": int(first_presolve_stats.group(4)),
                "Continuous Variables": int(first_presolve_stats.group(5)),
                "Total Constraints": int(first_presolve_stats.group(6)),
                "Presolving Time": float(first_presolve_stats.group(7)),
            }
            if "Total Presolving Time" not in stats or not stats["Total Presolving Time"]:
                stats["Total Presolving Time"] = float(first_presolve_stats.group(7))
            else:
                stats["Total Presolving Time"] += float(first_presolve_stats.group(7))

        for presolve in presolve_matches[1:]:
            presolve_stats = re.search(presolve_stats_pattern, presolve)
            if presolve_stats:
                stats["Other Presolve Stats"].append({
                    "Total Variables": int(presolve_stats.group(1)),
                    "Binary Variables": int(presolve_stats.group(2)),
                    "Integer Variables": int(presolve_stats.group(3)),
                    "Implicit Variables": int(presolve_stats.group(4)),
                    "Continuous Variables": int(presolve_stats.group(5)),
                    "Total Constraints": int(presolve_stats.group(6)),
                    "Presolving Time": float(presolve_stats.group(7)),
                })
                if "Total Presolving Time" not in stats or not stats["Total Presolving Time"]:
                    stats["Total Presolving Time"] = float(first_presolve_stats.group(7))
                else:
                    stats["Total Presolving Time"] += float(presolve_stats.group(7))

    # Extract total solving time
    solving_time_match = re.search(r"Solving Time \(sec\)\s*:\s*([\d.]+)", log_data)
    if solving_time_match:
        stats["Total Solving Time"] = float(solving_time_match.group(1))

    # Extract total nodes
    nodes_match = re.search(r"Solving Nodes\s*:\s*(\d+)", log_data)
    if nodes_match:
        stats["Total Nodes"] = int(nodes_match.group(1))

    # Extract primal bound
    primal_bound_match = re.search(r"Primal Bound\s*:\s*([-\d.e+]+)", log_data)
    if primal_bound_match:
        stats["Primal Bound"] = float(primal_bound_match.group(1))

    # Extract dual bound
    dual_bound_match = re.search(r"Dual Bound\s*:\s*([-\d.e+]+)", log_data)
    if dual_bound_match:
        stats["Dual Bound"] = float(dual_bound_match.group(1))

    # Extract gap
    gap_match = re.search(r"Gap\s*:\s*([\d.]+)\s*%", log_data)
    if gap_match:
        stats["Gap"] = float(gap_match.group(1))
    else:
        gap_match = re.search(r"Gap\s*:\s*infinite", log_data)
        if gap_match:
            # gap is infinite
            stats["Gap"] = float('inf')

    # Extracting problem details: number of variables and constraints
    var_pattern = r'presolved problem has (\d+) variables \((\d+) bin, (\d+) int, (\d+) impl, (\d+) cont\)'
    constraint_pattern = r'(\d+) constraints'
    type_pattern = r'(\d+) constraints of type <(\w+)>'

    # Extract the total number of variables and the number of each type
    var_match = re.search(var_pattern, log_data)
    if var_match:
        stats["Total Variables"] = int(var_match.group(1))
        stats["Binary Variables"] = int(var_match.group(2))
        stats["Integer Variables"] = int(var_match.group(3))
        stats["Implicit Variables"] = int(var_match.group(4))
        stats["Continuous Variables"] = int(var_match.group(5))
    else:
        if verbose:
            print("No variable information found in the text.")
        # write to error file
        # os.makedirs('errors', exist_ok=True)
        # current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # open(f'errors/error-{current_time_str}-no-var.txt', 'w').write(log_data)

    # Extract the total number of constraints
    constraint_match = re.search(constraint_pattern, log_data)
    if constraint_match:
        stats["Total Constraints"] = int(constraint_match.group(1))
    else:
        if verbose:
            print("No constraint information found in the text.")

    # Extract the number of constraints of each type
    type_matches = re.findall(type_pattern, log_data)
    stats["Constraint Types"] = {match[1]: int(match[0]) for match in type_matches}

    # Regex to extract each line of branch and bound process
    # pattern = r"""(?:[a-z]\s)?([\d\.]+s)\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^\|]+)\|\s*([^\|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)"""
    pattern = r"""(?:[a-z]\d?\s)?([\d\.]+s)\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)"""
    # pattern = r"""(?:[a-z]\s)?([\d\.]+s)\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^\|\s-]+|-)\s*\|\s*([^\|\s-]+|-)\s*\|\s*(\d+)\s*\|\s*([\d\w]+k?)\s*\|\s*([\d\w]+k?)\s*\|\s*([\d\w]+k?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d\.e\+]+)\s*\|\s*([\d\.e\+]+)\s*\|\s*([\d\.%]+)\s*\|\s*([^\|]+)"""
    bb_regex = re.compile(pattern, re.VERBOSE)
    bb_matches = [bb_regex.search(line) for line in log_data.strip().split('\n') if line.strip()]

    num_bb_nodes = 0
    num_bb_logs = 0
    first_lp_val = None
    root_time = None
    for match in bb_matches:
        if match:
            num_bb_logs += 1
            entry = match.groups()
            num_bb_nodes = max(num_bb_nodes, int(entry[1]))
            if detailed_logs:
                stats["BnB Logs"].append({
                    "Time": entry[0],
                    "Node": int(entry[1]),
                    "Left": int(entry[2]),
                    "LP Iterations": int(entry[3]),
                    "LP Iterations per Node": entry[4],
                    "Memory/Heuristic": entry[5],
                    "Max Depth": int(entry[6]),
                    "Variables": int(convert_value(entry[7])),
                    "Constraints": int(convert_value(entry[8])),
                    "Rows": int(convert_value(entry[9])),
                    "Cuts": int(entry[10]),
                    "Separations": int(entry[11]),
                    "Conflicts": int(entry[12]),
                    "Strong Branchings": int(entry[13]),
                    "Dual Bound": float(entry[14]),
                    "Primal Bound": float(entry[15]),
                    "Gap": float(entry[16].replace('%', '').strip()),
                    "Completeness": entry[17]
                })
            
            # Extract the first LP value
            if not first_lp_val and not match.group(0).strip()[0].isalpha():
                try:
                    first_lp_val = float(entry[14])
                except:
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('Error first LP')
                    os.makedirs('errors', exist_ok=True)
                    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    open(f'errors/error-{current_time_str}.txt', 'w').write(log_data)
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            
            # Extract the first non-root time
            if not root_time: 
                try:
                    if int(entry[1]) > 1:
                        root_time = float(entry[0][:-1])
                except:
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('Error first non root time')
                    os.makedirs('errors', exist_ok=True)
                    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    open(f'errors/error-{current_time_str}.txt', 'w').write(log_data)
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    stats["Number of BnB Logs"] = num_bb_logs

    # Extract the total number of nodes
    pattern = re.compile(r'Solving Nodes\s*:\s*(\d+)')
    match = pattern.search(log_data)
    if match:
        stats["Number of BnB Nodes"] = max(int(match.group(1)), num_bb_nodes)

    stats["First LP val"] = first_lp_val
    if first_lp_val is not None and stats["Primal Bound"] is not None:
        # if different sign, return very large value
        if stats["Primal Bound"] * first_lp_val < 0:
            stats["Integrality Gap"] = 1e6
        else:
            max_gap_val = max(abs(stats["Primal Bound"]), abs(first_lp_val))
            min_gap_val = min(abs(stats["Primal Bound"]), abs(first_lp_val))
            stats["Integrality Gap"] = max_gap_val / (min_gap_val + 1e-6) - 1
        # stats["Integrality Gap"] = 0 if stats["Primal Bound"] == 0 else abs((stats["Primal Bound"] - first_lp_val) / stats["Primal Bound"])

    stats["Root Time"] = root_time

    if stats["Gap"] == None:
        # save to error
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('Gap not found')
        os.makedirs('errors', exist_ok=True)
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        open(f'errors/error-{current_time_str}.txt', 'w').write(log_data)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    return stats


def parse_solving_process(log_file, detailed_logs=False):
    with open(log_file, 'r') as file:
        log_data = file.read()
    return parse_solving_process_from_content(log_data, detailed_logs=detailed_logs)
