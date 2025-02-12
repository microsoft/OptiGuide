import glob
import json
import os
import re
import time
import argparse
import random

from typing import List

import pandas as pd
import tqdm
from autogen.code_utils import extract_code
from contrast_extract_mps_info import analyze_mps, analyze_mps_high_level
from termcolor import colored
from utils import get_files_by_extension

from dotenv import load_dotenv
from openai import OpenAI

############################################################################
load_dotenv('../.env')
api_key = os.getenv('OPENAI_API_KEY', "")

client = OpenAI(
    api_key=api_key,
)

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
############################################################################


model_name = "gpt-4o"

char_prompt = """
Please comprehensively analyze the following mixed integer programming problem to identify its characteristics. 
Note that you should only include information that can be extracted from the MPS file, not the source code.

Present the results concisely in JSON format. 
The characteristics should include, but are not limited to, the following attributes:

- **MPS Format info:**: what the MPS file might contain.
- **Formulation Used:** (e.g., "big M", "inequality", or other formulation tricks)
- **Problem Domain:** (e.g., "bin packing", "set cover", or other popular operation research domains applied in the MIP)
- **Objective Function:** some details about the objective value
- **Constraints:** (e.g., "linear", "non-linear", etc.)
- **Variables:** (e.g., "integer", "binary", "continuous", etc.)
- **Other information:** 

---
{code}
"""

qa_prompt = """
Create a conversation between two optimization experts about a specific mixed integer programming instance (an MPS file). The conversation should focus on the mixed integer programming instance's data and numerical values, avoiding variable names, resource types, or other qualitative terms. The assistant's responses should be humble and uncertain, incorporating keywords like "might", "could," and "likely" due to the inability to see the exact source code or data.

Note: The USER knows nothing about this particular instance, while the ASSISTANT knows the following information.
```json
{characteristics}
```

The conversation should be insightful and avoid easy or superficial topics.

Format:
```
USER: xxx
ASSISTANT: yyy
```
"""

describe_prompt = """
The following information is extracted from a MPS file for mixed integer programming. Please generate the description of the MPS file to describe it:

--- Characteristics ---
{characteristics}


--- Some information from the MPS ---
{df_head}

--- Additional information from the MPS ---
{df_random}

--- Details ---
{mps_info}

--- Output ---

Use the JSON format:
```json
{{
    "information presented": "...",
    "description": "a paragraph that describes the MPS file and known information"
}}
```
"""


def get_description_for_mps(mps_filename: str, characteristic:str):
    try:
        # 2. Extract info from the MPS file
        mps_data = analyze_mps(mps_filename)

        df = pd.DataFrame.from_dict(mps_data, orient='index')
        df = df.transpose()
        df_head = df.head()
        df_random = df.sample(20)
        mps_info = analyze_mps_high_level(mps_filename)
    except Exception as e:
        print("Error while reading mps information for: ", mps_filename)
        print(colored(e, "red"))
        return ""

    prompt = describe_prompt.format(characteristics=characteristic,
                                    df_head=df_head,
                                    df_random=df_random,
                                    mps_info=mps_info)
    description = ""
    while description == "":
        try:
            description: str = do_oai_reply(prompt=prompt,
                                     model_name=model_name)
        except:
            time.sleep(10)
            continue

    details: str = extract_code(description)[-1][-1]
    description = ""
    try:
        description: str = json.loads(details)["description"]
    except Exception as e:
        print(e)
        print("Error parsing `description` from GPT-4's answer: ", details)

    if not description:
        try:
            description = re.findall(r'"description"\s*:\s*"(.*?)"',
                                        details)[0]
        except Exception as e:
            print(e)

    return description

def create_conv_for_file(code_filename: str,
                         outdir: str = "conv/",
                         n_mps=100,
                         provided_mps_files: List[str] = None):
    """
    Create conversation and description files for a given MPS file.

    This function processes a code file, extracts characteristics, and generates
    descriptions for associated MPS files. It can handle both provided MPS files
    and generate new ones if not provided.

    Args:
        code_filename (str): Path to the source code file.
        outdir (str, optional): Output directory for generated files. Defaults to "conv/".
        n_mps (int, optional): Number of MPS files to generate if not provided. Defaults to 100.
        provided_mps_files (List[str], optional): List of paths to existing MPS files. Defaults to None.

    Returns:
        None
    """
    os.makedirs(outdir, exist_ok=True)
    code = open(code_filename, "r").read()
    solve_code = re.findall(r"def solve\(.*?\):.*?return.*?\n", code,
                            re.DOTALL)[0]

    # Generate characteristics
    char_file = os.path.join(
        outdir,
        "chars_" + os.path.basename(code_filename).replace(".py", ".json"))

    if os.path.exists(char_file):
        characteristic = open(char_file, "r").read()
    else:
        chars: str = do_oai_reply(prompt=char_prompt.format(code=solve_code),
                                  model_name=model_name)
        characteristic: str = extract_code(chars)[-1][-1]
        open(char_file, "w").write(characteristic)

    # Generate description
    mps_files = provided_mps_files if provided_mps_files else range(n_mps)
    for mps_index, mps_file in tqdm.tqdm(enumerate(mps_files)):
        if provided_mps_files:
            out_mps_name = mps_file
        else:
            out_mps_name = os.path.join(
                outdir, f"seed_{mps_index}_" +
                os.path.basename(code_filename).replace('.py', '.mps'))

        out_mps_name = os.path.abspath(out_mps_name)
        out_desc_name = os.path.join(
            outdir,
            "desc_" + os.path.basename(out_mps_name).replace(".mps", ".txt"))

        if os.path.exists(out_desc_name):
            print("Skip existing description:", out_desc_name)
            continue

        if not os.path.exists(out_mps_name):
            # Generate MPS file only if not provided and doesn't exist
            _curr_code = code.replace(
                "model.optimize()",
                f"model.writeProblem('{out_mps_name}');')"
            )
            _curr_code = re.sub(r"seed = \d+", f"seed = {mps_index}", _curr_code)
            open("tmp.py", "w").write(_curr_code)

            try:
                os.system("timeout 60 python tmp.py")
                if not os.path.exists(out_mps_name):
                    continue  # timed out.
            except Exception as e:
                print("ERROR:", e)
                continue

        description = get_description_for_mps(mps_filename=out_mps_name, characteristic=characteristic)

        if description:
            open(out_desc_name, "w").write(str(description))

    return


def main_generate_seed_descriptions(parent_code_dir, parent_instance_dir, parent_output_dir):
    """
    Generate descriptions for seed MPS files.

    This function processes seed code files and their corresponding MPS files.
    It iterates through seed code files, finds the associated MPS files,
    and calls the create_conv_for_file function to generate descriptions.

    The function assumes a specific directory structure:
    - Seed code files are located in '{seed_root}/code/*.py'
    - MPS files are located in '{seed_root}/instances/{basename}/*.mps'

    The generated descriptions will be saved in the same directory as the MPS files.
    """
    # Handle Seed code and MPS
    seed_codes = glob.glob(os.path.join(parent_code_dir, "*.py"))
    seed_codes = sorted(seed_codes)

    for code_filename in seed_codes:
        glob_basename = os.path.basename(code_filename).replace(".py", "")

        mps_dir = os.path.join(parent_instance_dir, glob_basename)
        mps_names = get_files_by_extension(mps_dir)
        if len(mps_names) == 0: 
            print(colored(f"{mps_dir} is empty", "red"))
            continue
        
        for mps_name in mps_names:
            print("PROCESS:", code_filename, mps_name)
            create_conv_for_file(code_filename, 
                                outdir=os.path.join(parent_output_dir, glob_basename),
                                provided_mps_files=[mps_name, ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_code_dir', default='milp_code_v1/code', type=str)
    parser.add_argument('--parent_instance_dir', default='save_dir/instances/mps/code_v1', type=str)
    parser.add_argument('--parent_output_dir', default='save_dir/contrast/conv', type=str)
    args = parser.parse_args()
   
    parent_code_dir = args.parent_code_dir
    parent_instance_dir = args.parent_instance_dir
    parent_output_dir = args.parent_output_dir

    main_generate_seed_descriptions(parent_code_dir, parent_instance_dir, parent_output_dir)