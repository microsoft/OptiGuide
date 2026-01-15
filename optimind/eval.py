#!/usr/bin/env python3

import asyncio, json, logging, os, re, sys, time, random
import pandas as pd, numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from json_repair import repair_json
import nest_asyncio, pdb
from datetime import datetime
from collections import Counter
import argparse, os, logging, warnings, subprocess, math, random, ast, re
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable, NamedTuple
import math
import copy

from eval_utils.utils_general import (
    setup_model_and_env,
    as_list, 
    get_hint_instructions, 
    load_ea_pairs, 
    build_error_analysis_str, 
)
from eval_utils.build_prompt import (
    build_prompt_optmath,
    build_prompt_sirl,
    build_prompt_sirl_system,
)

# ────────────────── ARGUMENT PARSER ─────────────────────────────────
parser = argparse.ArgumentParser(description="Run multi-turn evaluation with hybrid backend (SGLang or openai).")
# data name
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Path to the model.")
parser.add_argument("--data", type=str, required=True, help="Path to a .csv, .jsonl, or .parquet file to evaluate.")
parser.add_argument("--max-turns", type=int, default=5, help="maximum number of turns")
parser.add_argument("--num-majority", type=int, default=1, help="Number of parallel samples per turn for majority vote. 1 = no voting.")


# Use cached model only
parser.add_argument("--offline_only", action="store_true",
                    help="Force fully-offline mode (error if files are not in local cache).")
parser.add_argument("--hf_cache_dir", type=str, default=None,
                    help="Path to HF cache (default: ~/.cache/huggingface). If set, we only read from here.")
parser.add_argument("--hf_revision", type=str, default=None,
                    help="Specific revision/commit to load from local cache (optional, default fetching the newest snapshot from local cache).")

# backend params
parser.add_argument("--backend_str", type=str, default="sglang", help="Backend to use: hybrid, sglang, openai.")
parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for SGLang.")
parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size for SGLang.")
parser.add_argument("--suffix", type=str, default="", help="Suffix for the output directory.")
parser.add_argument("--openapi_api_key_name", type=str, default="OPENROUTER_API_KEY", help="Environment variable name for OpenAI API key.")
parser.add_argument("--openai_base_url", type=str, default="https://openrouter.ai/api/v1", help="Base URL for OpenAI SDK when backend_str=openai.")
parser.add_argument("--openai_model_name", type=str, default="x-ai/grok-4.1-fast", help="OpenAI SDK model name (e.g. 'x-ai/grok-4.1-fast') when backend_str=openai.")
parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility.")
parser.add_argument("--batch_size_sglang", type=int, default=256, help="Batch size for sglang.")

# sampling params
parser.add_argument("--temp", type=float, default=0.6, help="Temperature for sampling.")
parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling.")

# gpt oss params
parser.add_argument("--gpt-oss", action="store_true", help="use gpt oss based model")
parser.add_argument("--reasoning-effort", type=str, default="medium", help="gpt oss reasoning effort")

# prompt params
parser.add_argument("--user_prompt_type", default="default", type=str, choices=["default", "optmath", "sirl"],
                    help="Type of prompt to use (default, optmath, sirl.).")
parser.add_argument("--tool_prompt_type", default="default", type=str, choices=["default", "add_question", "add_hint", "add_initial_prompt"],
                    help="Type of prompt to use (default, optmath, sirl.).")
parser.add_argument("--system_prompt_type", default="default", type=str, choices=["default", "simple"])
parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist.")
parser.add_argument("--debug", action="store_true", help="Debug the code.")


# error analysis params
parser.add_argument("--apply_error_analysis", action="store_true", help="Use error analysis in the prompt")
parser.add_argument("--max_edit", type=int, default=5,
               help="Max edit distance for fuzzy matching 'problem_class' to error-analysis types.")
parser.add_argument("--error_analysis_file", type=str, help="CSV with error analysis & hints.")
parser.add_argument("--apply_hint_instructions", action="store_true", help="Add instructions for using hints and general hints")

# early stopping in multi-turn correction
parser.add_argument("--no_early_stop", action="store_true", help="Do not early terminate if the model outputs a correct answer before max turns.")

# majority voting
parser.add_argument("--majority-rtol", type=float, default=1e-6, help="Relative tolerance for majority voting.")
parser.add_argument("--majority-atol", type=float, default=1e-4, help="Absolute tolerance for majority voting.")
parser.add_argument("--majority_vote_final", action="store_true", help="Use majority vote to select the final answer among all turns.")

# Utils
parser.add_argument("--precision", type=float, default=1e-6, help="Precision for comparing float answers.")
parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory to save results.")

# parse arguments
args = parser.parse_args()

# ────────────────── NEST ASYNCIO ────────────────────────────────────
nest_asyncio.apply()
asyncio.set_event_loop(asyncio.new_event_loop())

# ───────────────────────── CONFIG ────────────────────────────────────
cfg_model = setup_model_and_env(args)
MODEL_PATH = cfg_model["MODEL_PATH"]
MODEL_NAME = cfg_model["MODEL_NAME"]
MODEL_LOAD_PATH = cfg_model["MODEL_LOAD_PATH"]
        
DATA_PATH          = args.data
MAX_TURNS          = args.max_turns
NUM_MAJORITY       = args.num_majority
MAJORITY_FINAL     = args.majority_vote_final
MAX_NEW_TOKENS     = 8_000 if not args.gpt_oss else 24_000  # qwen is subject to context limit
TEMP, TOP_P        = args.temp, args.top_p      
BATCH_SIZE_SGLANG  = args.batch_size_sglang       # prompts per GPU pass
THREADS_POOL       = 60                        # parallel api calls
MAX_TOOL_PAR       = 60                      # concurrent CodeExecutionTool
TP_SIZE            = args.tp_size
DP_SIZE            = args.dp_size

# gpt oss is not subject to this token limit, so only for qwen
MAX_PROMPT_TOKEN   = 24_000    # a few tokens for assistant message
PRECISION          = args.precision

if args.backend_str not in ["hybrid", "sglang", "openai"]:
    BACKEND_SCHEDULE = ["sglang"] * MAX_TURNS    
    BACKEND_STR      = "sglang" if all(b == "sglang" for b in BACKEND_SCHEDULE) else "hybrid"
else:
    BACKEND_SCHEDULE = ["sglang"] + ["openai"] * (MAX_TURNS - 1) if args.backend_str == "hybrid" else \
                        [args.backend_str] * MAX_TURNS
    BACKEND_STR      = args.backend_str
   
TOOLS = [{
    "type": "function",
    "function": {
        "name": "code_execution",
        "description": "Execute Gurobi code and return stdout / stderr",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "The gurobipy code to execute"}},
                       "required": ["code"]},
        "strict": False,
    },
}]
CODE_TOOL_IDX = 0

# ───────────────── CodeExecutionTool ─────────────────────────────────
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
print("Load parent path", path)

from recipe.tools.code_execution import CodeExecutionTool
from rewards.rm_objective_gurobipy_full import compute_score
from rewards.utils import  compute_score_history_output
from rewards.utils import extract_python_code_block as extract_python_code_block_helper

from recipe.tools.schemas import OpenAIFunctionToolSchema

exec_tool = CodeExecutionTool({"max_assistant_turns": MAX_TURNS, "tool_format": "interaction"},
                               OpenAIFunctionToolSchema.model_validate(TOOLS[CODE_TOOL_IDX]))
tool_sem = asyncio.Semaphore(MAX_TOOL_PAR)


####################### TOOL PROMPT and SYSTEM PROMPT ##########################
CODE_USER_STR = """
Running the given gurobipy code block result in the following standard output and error. 
Please analyze the standard output and error and provide your thoughts on the correctness of the code and the output, and if it is not correct, provide a corrected version of the code.

{tool_output}

# Note:
- The Code must include:```python
import gurobipy as gp
from gurobipy import GRB
```
- Make sure the model variable is named `model`.
- Avoid using "<" and ">" in Gurobi constraints; instead, use "<=" or ">=".
- Carefully determine whether the variable is an integer or continuous.
"""

CODE_SYSTEM_PROMPT = """You are an expert in optimization and mixed integer programming. You are given an optimization problem and you need to solve it using gurobipy. 
Reasoning step by step before generating the gurobipy code. The user will execute the code and return the standard output and error to you, and you should use the user response to refine your previously generated code.
When you are satisfied with the code, you can put your final answer in the format of `#### <answer>`."""

SIMPLE_SYSTEM_PROMPT = """You are an expert in optimization and mixed integer programming. You are given an optimization problem and you need to solve it using gurobipy. 
Reasoning step by step before generating the gurobipy code. """



# ─────────────────── Build prompt with error analysis ───────────────────────────────────────────────
def build_first_turn_user_prompt_list(question: str, error_pairs: List = []) -> str:
    lines = []
    lines.append("You are an expert in optimization and mixed integer programming. You are given an optimization problem and you need to solve it using gurobipy.\n")
    
    lines.append(question.strip())
        
    local_hint, global_hint = "", ""
    if len(error_pairs) > 0:
        local_hint = build_error_analysis_str(error_pairs)
        lines.append(local_hint)
    
    if args.apply_hint_instructions:
        global_hint = get_hint_instructions()
        lines.append(global_hint)
        
    combined_hint = "" if not (local_hint or global_hint) else "\n".join([local_hint, global_hint])
    
    lines.append(
        "Reason step by step before generating the gurobipy code.\n"
        "When you respond, first think carefully.\n"
        "After thinking, output the math modeling of the problem.\n"
        "Finally output a ```python ...``` code block that solves the problem.\n"
        "The code must include:\n"
        "import gurobipy as gp\n"
        "from gurobipy import GRB\n"
    )

    print(combined_hint)
    return "".join(lines), combined_hint

def extract_python_code_block(response: str) -> str | None:
    return extract_python_code_block_helper(response, return_none_if_failed=True)

# helper function to get more tool prompt if needed
def get_more_tool_prompt(item):
    if args.tool_prompt_type == "default":
        return ""
    
    if args.tool_prompt_type == "add_question":
        more_tool_prompt = f"""Recap: the optimization problem that you need to solve using gurobipy is 
{item['initial_question']}""" 
    elif args.tool_prompt_type == "add_hint":
        more_tool_prompt = f"""Recap: {item['initial_hint'].lstrip()}"""
    elif args.tool_prompt_type == "add_initial_prompt":
        more_tool_prompt = f"""Recap: the optimization problem that you need to solve using gurobipy is
{item['initial_question']}
{item['initial_hint'].lstrip()}
"""
    
    return more_tool_prompt

async def run_tool_as_user(item, call_arg, more_tool_prompt=""):
    if call_arg is None or not call_arg:
        if len(item["metrics"]) == 0 or args.no_early_stop:
            # in these cases, we may ask the model to generate new code; 
            item["history"].append({"role": "user", "content": "No python code block found. Skipping code execution. If you believe the last provided code is correct, you can stop. Otherwise, please provide a corrected code block."})
            
            # We default the metric to the previous round if it exists. 
            # However, if valid initial code is extracted, we have no previous solution to default to, hence we set all metrics to 0.0
            if len(item["metrics"]) == 0:
                item["metrics"].append({"score": 0.0, "no_execution_error":  0.0, 
                                        "have_objective": 0.0, "objective": 0.0, "accuracy": 0.0})     
            else:
                item["metrics"].append(copy.deepcopy(item["metrics"][-1]))      
        return
    
    async with tool_sem:
        out, _, m = await exec_tool.execute(item["iid"], call_arg, precision=PRECISION)
    
    
    user_output = CODE_USER_STR.format(tool_output=out)
    if more_tool_prompt:
        user_output += "\n" + more_tool_prompt
        
    item["history"].append({"role": "user", "content": user_output})
    
    item["metrics"].append(m)
    item["found_code"] = True

### find a way to convert the assistant history into a tool call history by splitting the text and the code and use that for tool call

# ─────────────────── Tokenize prompts ────────────────────────────────
from transformers import AutoTokenizer

# if global variable tokenizer is not set, load the tokenizer
if 'tokenizer' not in globals():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_LOAD_PATH,
        trust_remote_code=True,
        local_files_only=args.offline_only,   # <-- offline only if requested
    )
else:
    tokenizer = globals()['tokenizer']

  
if args.gpt_oss:
    from eval_utils.harmony_utils import harmony_parse_calls

def to_prompt_no_tools(msgs, add_generation_prompt=True):
    if args.backend_str in ["azure", "openai"]:
        # TOOLS are encoded in the openai query api, not by the string template
        # concatenate all messages directly, assuming no tokenizer
        segs_str = "\n".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in msgs)
        if add_generation_prompt:
            segs_str += "Assistant: "
        return segs_str
    
    
    segs_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, tokenize=False)
    
    if args.gpt_oss and args.reasoning_effort != "medium":
        # search for "Reasoning: (low | mid | high)", replace the first
        segs_str = re.sub(r"(Reasoning:\s*)(low|medium|high)", 
                            lambda m: m.group(1) + args.reasoning_effort, segs_str, count=1, flags=re.IGNORECASE)

    # cap the prompt window for qwen models
    if not args.gpt_oss:  
        segs_tokens = tokenizer.encode(segs_str)
        if len(segs_tokens) > MAX_PROMPT_TOKEN:
            print(f"Warning: the prompt exceeds the maximum token length, truncating to fit (Current length {len(segs_tokens)} | Length limit {MAX_PROMPT_TOKEN}).")
            segs_str = tokenizer.decode(segs_tokens[-MAX_PROMPT_TOKEN:])

    return segs_str

def majority_vote_select(answers: List[Optional[float]], rtol: float = 1e-6, atol: float = 1e-4) -> Tuple[int, Optional[float], int]:
    clusters: List[Tuple[float, List[int]]] = []
    for idx, val in enumerate(answers):
        if val is None:
            continue
        placed = False
        for rep, idxs in clusters:
            if math.isclose(float(val), float(rep), rel_tol=rtol, abs_tol=atol):
                idxs.append(idx); placed = True; break
        if not placed:
            clusters.append((float(val), [idx]))
    if not clusters:
        return 0, None, 0
    clusters.sort(key=lambda x: (-len(x[1]), min(x[1])))
    rep_val, idxs = clusters[0]
    selected_j = min(idxs)
    return selected_j, rep_val, len(idxs)

##### General 
class GenOutput(NamedTuple):
    text: str
    token_ids: Optional[List[int]] = None  # only for sglang 

# type alias: async function (histories, pool, loop) -> list[GenOutput]
BackendGenFn = Callable[
    [List[List[Dict[str, Any]]], ThreadPoolExecutor, asyncio.AbstractEventLoop],
    Awaitable[List[GenOutput]],
]

async def generic_gen(
    todo_items: List[Dict[str, Any]],
    backend_label: str,
    generate_batch: BackendGenFn,
    pool: ThreadPoolExecutor,
    loop: asyncio.AbstractEventLoop,
):
    """
    Shared generation logic for all backends.

    - `generate_batch(histories)` returns List[GenOutput] with same length.
    - `histories` is a list of chat histories: List[{"role", "content"}].
    """
    print(f"Generating with {backend_label}...")

    # ---------- 1) one or many samples per item ----------
    if MAJORITY_FINAL or NUM_MAJORITY == 1:
        histories = [it["history"] for it in todo_items]
        outs = await generate_batch(histories, pool, loop)
    else:
        histories = [it["history"] for it in todo_items for _ in range(NUM_MAJORITY)]
        repeated_outs = await generate_batch(histories, pool, loop)

        outs = []
        for idx, it in enumerate(todo_items):
            outs_idx = repeated_outs[idx * NUM_MAJORITY:(idx + 1) * NUM_MAJORITY]

            repeated_objectives_idx = []
            for o in outs_idx:
                res = compute_score("nl4ilp", o.text, it["ground_truth"], extra_info=None, precision=PRECISION)
                obj = res["objective"] if res["have_objective"] else None
                repeated_objectives_idx.append(obj)

            if repeated_objectives_idx:
                chosen_j, majority_objective, majority_count = majority_vote_select(
                    repeated_objectives_idx, rtol=args.majority_rtol, atol=args.majority_atol,
                )
                chosen = outs_idx[chosen_j] if majority_objective is not None else outs_idx[0]
            else:
                chosen = outs_idx[0]

            outs.append(chosen)

    # ---------- 2) common postprocessing: history + code + calls_args ----------
    for it, o in zip(todo_items, outs):
        txt = o.text
        token_ids = o.token_ids

        if args.gpt_oss and token_ids is not None:
            # only for gpt-oss models (sglang path)
            reasoning_content, final_content = harmony_parse_calls(tokenizer, token_ids, txt, keep_tools=False)
            code = extract_python_code_block(txt)
                
            if reasoning_content and final_content:
                it["history"].append({"role": "assistant", "content": final_content, "thinking": reasoning_content})
            elif reasoning_content or final_content:
                content = reasoning_content if reasoning_content else final_content
                it["history"].append({"role": "assistant", "content": content})
            else:
                it["history"].append({"role": "assistant", "content": txt})
        else:
            code = extract_python_code_block(txt)
            it["history"].append({"role": "assistant", "content": txt})

        if not code:
            print(f"No code block found in {backend_label} response, skipping item.")
            it["calls_args"] = []
        else:
            # we store the full reply as "code" so the tool can parse the python block inside
            it["calls_args"] = [{"code": txt}]

    print(f"{backend_label} generation done.")


# ─────────────────── S G L A N G ─────────────────────────────────────
if "sglang" in BACKEND_SCHEDULE:
    import sglang as sgl
    
    async def sglang_generate_batch(histories, pool, loop) -> List[GenOutput]:
        prompts = [to_prompt_no_tools(h) for h in histories]
        outs = await engine.async_generate(prompts, sglang_sampling_params)

        rets: List[GenOutput] = []
        for o in outs:
            text = o["text"]
            token_ids = o.get("output_ids") if args.gpt_oss else None
            rets.append(GenOutput(text=text, token_ids=token_ids))
        return rets


# ──────────────────── O P E N A I  (remote backend) ─────────
if "openai" in BACKEND_SCHEDULE:
    from openai import OpenAI
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "eval_utils/.env"))

    OPENAI_BASE_URL = args.openai_base_url  # e.g. "https://openrouter.ai/api/v1"
    OPENAI_MODEL = args.openai_model_name 
    OPENAI_API_KEY = os.environ.get(args.openapi_api_key_name)
    
    if not OPENAI_API_KEY:
        raise RuntimeError(
            f"{args.openapi_api_key_name} environment variable is not set. "
            f"Export it before running when backend_str=openai."
        )
   
    openai_client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    def _openai_block(prompt, client, model_name):
        if not isinstance(prompt, list):
            prompt = [{"role": "user", "content": prompt}]

        m = None
        for _ in range(5):
            try:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    # temperature=TEMP,
                    # top_p=TOP_P,
                )
                response = r.choices[0].message
                m = response.content
            except Exception as e:
                print(f"OpenAI call error: {e}, retrying...")
                import pdb; pdb.set_trace()
                time.sleep(random.uniform(2, 5))
        # import pdb; pdb.set_trace()
        return m if m else ""
    
    async def openai_generate_batch(histories, pool, loop) -> List[GenOutput]:
        replies = await asyncio.gather(
            *[loop.run_in_executor(pool, partial(_openai_block, h, client=openai_client,
                                                 model_name=OPENAI_MODEL)) for h in histories])
        return [GenOutput(text=reply) for reply in replies]
            
                
# ──────────────────── hybrid dispatch ───────────────────────────────
async def dispatch_backend(todo_items, backend, pool, loop):
    if backend == "sglang":
        await generic_gen(todo_items, backend_label="SGLANG",
            generate_batch=sglang_generate_batch, pool=pool, loop=loop)
    elif backend == "openai":
        await generic_gen(todo_items, backend_label=f"OpenAI {OPENAI_MODEL} at {OPENAI_BASE_URL}",
            generate_batch=openai_generate_batch, pool=pool, loop=loop)
    else:
        raise ValueError(backend)
  

# ──────────────────── Solve batch ────────────────────────────────────
from eval_utils.build_prompt import build_prompt_optmath, build_prompt_sirl, build_prompt_sirl_system

def get_assistant_messages(history):
    assistants = [d for d in history if isinstance(d, dict) and d.get("role") == "assistant"]
    sol_assistant = "\n".join([x for a in assistants for x in [a.get("thinking", ""), a.get("content", "")] if x])
    return sol_assistant

def get_all_messages(history):
    sol_str = "\n".join([x for d in history for x in [d.get("thinking", ""), d.get("content", ""), "-"*30] if x])
    return sol_str

            

async def final_compute_score(items, pool, loop):
    sol_extracted = [None] * len(items)
    
    recs = []
    for it, sol_ex in zip(items, sol_extracted):
        if it["history"][0]["role"] == "system":
            pstr = to_prompt_no_tools(it["history"][:2], add_generation_prompt=False)
            fstr = to_prompt_no_tools([h for i, h in enumerate(it["history"]) if i < 2 or h["role"] == "assistant"], add_generation_prompt=False)
        else:
            pstr = to_prompt_no_tools(it["history"][:1], add_generation_prompt=False)
            fstr = to_prompt_no_tools([h for i, h in enumerate(it["history"]) if i == 0 or h["role"] == "assistant"], add_generation_prompt=False)
        pl   = len(tokenizer.encode(pstr,  add_special_tokens=False))
        sl   = len(tokenizer.encode(fstr[len(pstr):],   add_special_tokens=False))
        
        res = compute_score_history_output("nl4ilp", it["history"], it["ground_truth"], extra_info=None, precision=PRECISION)

        it.update(objective=res["objective"], score=res["accuracy"], 
                no_execution_error=res["no_execution_error"],
                have_objective=res["have_objective"],
                prompt_len=pl, solution_len=sl)
 
        ##### save record
        for t in range(MAX_TURNS):
            it[f"accuracy_{t}"] = (it["metrics"][t]["accuracy"] if t < len(it["metrics"]) else it.get(f"accuracy_{t-1}", 0.0))
            it[f"objective_{t}"] = (it["metrics"][t]["objective"] if t < len(it["metrics"]) else it.get(f"objective_{t-1}", 0.0))
            it[f"no_execution_error_{t}"] = (it["metrics"][t]["no_execution_error"] if t < len(it["metrics"]) else it.get(f"no_execution_error_{t-1}", True))
            it[f"have_objective_{t}"] = (it["metrics"][t]["have_objective"] if t < len(it["metrics"]) else it.get(f"have_objective_{t-1}", False))
            
        if MAX_TURNS > 0 and it[f"accuracy_{MAX_TURNS-1}"] != res["accuracy"]:
            save_idx = 0
            os.makedirs("tmp_compute_score", exist_ok=True)
            while os.path.exists(f"tmp_compute_score/item_{save_idx}.json"):
                save_idx += 1
            with open(f"tmp_compute_score/item_{save_idx}.json", "w") as fp:
                json.dump({"history": it["history"], "ground_truth": it["ground_truth"], 
                           f"accuracy_{MAX_TURNS-1}": it[f"accuracy_{MAX_TURNS-1}"], "accuracy": res["accuracy"]}, fp)
            with open(f"tmp_compute_score/history_{save_idx}.txt", "w") as f:
                f.write(get_all_messages)
            
        recs.append({"id": it["iid"],
                    "objective": it["objective"],
                    "ground_truth": it["ground_truth"],
                    "score": it["score"],
                    "no_execution_error": it["no_execution_error"],
                    "have_objective": it["have_objective"],
                    "accuracy_per_turn": {f"accuracy_{t}": it[f"accuracy_{t}"] for t in range(MAX_TURNS)},
                    "objective_per_turn": {f"objective_{t}": it[f"objective_{t}"] for t in range(MAX_TURNS)},
                    "no_execution_error_per_turn": {f"no_execution_error_{t}": it[f"no_execution_error_{t}"] for t in range(MAX_TURNS)},
                    "have_objective_per_turn": {f"have_objective_{t}": it[f"have_objective_{t}"] for t in range(MAX_TURNS)},
                    "dialogue": it["history"],
                    "prompt_len": it["prompt_len"],
                    "solution_len": it["solution_len"]})
    
    return recs
  

async def build_items(rows, majority_final=MAJORITY_FINAL, num_majority=NUM_MAJORITY):
    items = []
    ##################### load error analysis prompt #####################
    # (Optional) early file existence check
    if args.apply_error_analysis and args.error_analysis_file and not Path(args.error_analysis_file).exists():
        raise FileNotFoundError(f"error_analysis_file not found: {args.error_analysis_file}")

    if args.apply_error_analysis:
        # read error analysis file
        ea = pd.read_csv(Path(args.error_analysis_file))
        # define once (removed duplicate)
        class_to_pairs = load_ea_pairs(ea, args)

    ##################### load system (optional) and user prompts, build items #####################
    for ir, r in enumerate(rows):
        question = getattr(r, "question", None)
        answer = getattr(r, "answer", None)
        answer = ast.literal_eval(answer) if isinstance(answer, str) else answer
        classes = as_list(getattr(r, "problem_class", []))
        combined_hint = ""
        
        if args.apply_error_analysis and classes:
            error_pairs: List[Tuple[str, str]] = []
            for c in classes:
                error_analyses = class_to_pairs[c] if c in class_to_pairs else []
                error_pairs.extend(error_analyses)

            # row-level dedup (type corrected)
            seen = set()
            dedup: List[Tuple[str, str]] = []
            for a in error_pairs:
                key = a
                if key not in seen:
                    seen.add(key)
                    dedup.append(key)

            if args.user_prompt_type == "optmath":
                user_content = build_prompt_optmath(question) + "\n" + build_error_analysis_str(dedup)
            elif args.user_prompt_type == "sirl":
                user_content = build_prompt_sirl(question) + "\n" + build_error_analysis_str(dedup)
            else:
                try:
                    user_content, combined_hint = build_first_turn_user_prompt_list(question, dedup)
                except Exception as e: 
                    import pdb; pdb.set_trace()
        else:
            if args.user_prompt_type == "optmath":
                user_content = build_prompt_optmath(question)
                if args.apply_hint_instructions:
                    user_content += "\n" + get_hint_instructions()
            elif args.user_prompt_type == "sirl":
                user_content = build_prompt_sirl(question)
                if args.apply_hint_instructions:
                    user_content += "\n" + get_hint_instructions()
            else:
                try:
                    user_content, combined_hint = build_first_turn_user_prompt_list(question)
                except Exception as e: 
                    import pdb; pdb.set_trace()

        if args.user_prompt_type == "sirl":
            # If build_prompt_sirl_system has parameters, pass them here instead.
            system_content = build_prompt_sirl_system()
        else:
            system_content = CODE_SYSTEM_PROMPT if args.system_prompt_type == "default" else SIMPLE_SYSTEM_PROMPT

        if args.no_early_stop and args.user_prompt_type != "sirl":
            hist = [{"role": "user", "content": user_content}]
        else:
            hist = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ]

        if majority_final and num_majority > 1:
            for i_majority in range(num_majority):
                iid = await exec_tool.create(instance_id=ir * num_majority + i_majority, ground_truth=answer)
                it = {"iid": iid, "initial_question": question,
                      "ground_truth": answer, "ground_truth_code": getattr(r, "code", None),
                      "history": copy.deepcopy(hist), "metrics": [], "done": False, "found_code": False}
                if args.tool_prompt_type != "default":
                    it.update({"initial_hint": combined_hint})
                items.append(it)
        else:
            iid = await exec_tool.create(instance_id=ir, ground_truth=answer)
            it = {"iid": iid, "initial_question": question, 
                  "ground_truth": answer, "history": hist, "metrics": [], "done": False, "found_code": False}
            if args.tool_prompt_type != "default":
                it.update({"initial_hint": combined_hint})
            items.append(it)
            
    return items


async def solve_batch_no_tools(rows):
    items = await build_items(rows)

    ##################### main loop #####################
    pool = ThreadPoolExecutor(max_workers=THREADS_POOL)
    loop = asyncio.get_running_loop()
    
    # if majority_final: repeat the items and then aggregate at the end 

    print(f"Data: {DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Seed: {args.seed}")
    print(f"OUT_DIR: {OUT_DIR}")
    
    print(f"Error analysis file: {args.error_analysis_file}")
    for turn in range(MAX_TURNS):
        print("Start turn", turn + 1)
        backend = BACKEND_SCHEDULE[min(turn, len(BACKEND_SCHEDULE)-1)]
        todo = [it for it in items if not it["done"]]
        if not todo:
            break

        await dispatch_backend(todo, backend, pool, loop)
        
        print("Finished llm generation for turn", turn + 1)
        
        calls_args_list = [it["calls_args"][0] if "calls_args" in it and len(it["calls_args"]) > 0 else None for it in todo]
        tool_tasks = [run_tool_as_user(it, calls_args,  
                                       more_tool_prompt=get_more_tool_prompt(it) if args.tool_prompt_type != "default" else "") 
                      for it, calls_args in zip(todo, calls_args_list)]  
        if tool_tasks:
            await asyncio.gather(*tool_tasks)
        
        accuracies_turn = [it["metrics"][-1]["accuracy"] if ("metrics" in it and len(it["metrics"]) > 0) else 0.0 for it in items]
        print("Finished tool calls for turn", turn + 1, f"accuracy: {np.mean(accuracies_turn)} +- {np.std(accuracies_turn)})")

        for it in todo: 
            # Note: calls_args is empty if no code is extracted
            if args.no_early_stop or not it["found_code"]: 
                it["done"] = False  # we never done until max turns | or if no code has been found yet, we continue
            else:
                # stop if cannot find any code; condition in some previous turn has found some code
                it["done"] = len(it["metrics"]) > 0 and "calls_args" in it and len(it["calls_args"]) == 0
                
    pool.shutdown(wait=True)
    await asyncio.gather(*(exec_tool.release(it["iid"]) for it in items))
    
    if MAJORITY_FINAL and NUM_MAJORITY > 1:
        # aggregate the results for each group of NUM_MAJORITY items using majority_vote_select
        aggregated_items = []
        for i in range(0, len(items), NUM_MAJORITY):
            group = items[i:i+NUM_MAJORITY]
            # aggregate base on the objectives
            group_objectives = [
                it["metrics"][-1]["objective"]
                for it in group if ("metrics" in it and len(it["metrics"]) > 0 and it["metrics"][-1]["have_objective"])
            ]
            if group_objectives:
                chosen_j, majority_objective, majority_count = majority_vote_select(group_objectives, rtol=args.majority_rtol, atol=args.majority_atol)
                chosen = group[chosen_j] if majority_objective is not None else group[0]
            else:
                majority_objective = None
                chosen = group[0]
            
            # build the aggregated item
            aggregated_item = {
                "iid": chosen["iid"] // NUM_MAJORITY,
                "ground_truth": chosen["ground_truth"],
                "ground_truth_code": chosen.get("ground_truth_code", None),
                "history": chosen["history"],
                "metrics": chosen["metrics"],
                "done": chosen["done"],
            }
            aggregated_items.append(aggregated_item)
            # print("Aggregated item from iids:", [it["iid"] for it in group], "to iid:", aggregated_item["iid"], "with objective:", majority_objective)
            
        items = aggregated_items

    # final compute score
    pool = ThreadPoolExecutor(max_workers=THREADS_POOL)
    loop = asyncio.get_running_loop()
    recs = await final_compute_score(items, pool, loop)
    pool.shutdown(wait=True)
    
    return recs


# ────────────────────── DRIVER ───────────────────────────────────────
async def main():
    ext = os.path.splitext(DATA_PATH)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(DATA_PATH)
    elif ext in (".jsonl", ".json"):
        # jsonl: one JSON object per line
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(recs)
    else:
        # fallback keeps previous behavior for .parquet
        df = pd.read_parquet(DATA_PATH)
        
    # drop rows where question is nan or answer is nan or question not in df or answer not in df
    df = df.dropna(subset=["question", "answer"])
    df = df.reset_index(drop=True)
    # only load first 10 examples for debugging | remember to set back!
    print(f"Loaded {len(df)} valid records from {DATA_PATH}")
        
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)   
            
    all_recs = []
    for i in range(0, len(df), BATCH_SIZE_SGLANG):
        recs = await solve_batch_no_tools(df.iloc[i:i+BATCH_SIZE_SGLANG].itertuples(index=False))
        # import pdb; pdb.set_trace()
        with open(OUT_PATH, "a") as fh:
            for idx, rec in enumerate(recs):
                rec["id"] = idx + i  # fix the index
                fh.write(json.dumps(rec, indent=4) + "\n")
        all_recs.extend(recs)
        # import pdb; pdb.set_trace()
    
    # also write to .json
    with open(OUT_PATH.replace(".jsonl", ".json"), "w") as f:
        json.dump(all_recs, f, indent=4)    
    
    print("Results write to:", OUT_PATH, "and ", OUT_PATH.replace(".jsonl", ".json"))

    # quick stats
    scores = [rec["score"] for rec in all_recs]
    no_execution_errors = [rec["no_execution_error"] for rec in all_recs]
    have_objectives = [rec["have_objective"] for rec in all_recs]
    plens  = [rec["prompt_len"] for rec in all_recs]
    slens  = [rec["solution_len"] for rec in all_recs]
    print(f"Avg prompt: {np.mean(plens):.1f} tok | Avg response: {np.mean(slens):.1f} tok")
    print(f"Avg accuracy: {100*np.mean(scores):.2f}% (n={len(scores)})")
    print(f"Avg no execution error: {100*np.mean(no_execution_errors):.2f}% (n={len(no_execution_errors)})")
    print(f"Avg have objective: {100*np.mean(have_objectives):.2f}% (n={len(have_objectives)})")
    for t in range(MAX_TURNS):
        ts = [rec["accuracy_per_turn"][f"accuracy_{t}"] for rec in all_recs]
        print(f"Turn {t}: {100*np.mean(ts):.2f}%")
    
    # save the above stats in stats.json
    stats = {
        "num_items": len(all_recs),
        "avg_prompt_len": np.mean(plens),
        "avg_solution_len": np.mean(slens),
        "avg_accuracy": np.mean(scores),
        "avg_no_execution_error": np.mean(no_execution_errors),
        "avg_have_objective": np.mean(have_objectives),
        "accuracy_per_turn": {f"turn_{t}": np.mean([rec["accuracy_per_turn"][f"accuracy_{t}"] for rec in all_recs]) for t in range(MAX_TURNS)},
        "no_execution_error_per_turn": {f"turn_{t}": np.mean([rec["no_execution_error_per_turn"][f"no_execution_error_{t}"] for rec in all_recs]) for t in range(MAX_TURNS)},
        "have_objective_per_turn": {f"turn_{t}": np.mean([rec["have_objective_per_turn"][f"have_objective_{t}"] for rec in all_recs]) for t in range(MAX_TURNS)},
    }
    
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    with open(os.path.join(OUT_DIR, "backend_schedule.json"), "w") as f:
        json.dump(BACKEND_SCHEDULE, f)
        

if __name__ == "__main__":  
    # if error analysis assert error_analysis_file is given
    if args.apply_error_analysis and not args.error_analysis_file:
        raise ValueError("error_analysis_file must be provided if apply_error_analysis is set.")
    
    if not args.apply_error_analysis and not args.apply_hint_instructions:
        assert args.tool_prompt_type in ("default", "add_question"), "If not applying hint instructions or error analysis, tool_prompt_type must be 'default' or 'add_question'."
    

    SYSTEM_PROMPT_TYPE_STR = "" if args.system_prompt_type == "default" else f"_{args.system_prompt_type}"
    USER_PROMPT_TYPE_STR = "" if args.user_prompt_type == "default" else f"_{args.user_prompt_type}"
    TOOL_PROMPT_TYPE_STR = "" if args.tool_prompt_type == "default" else f"_{args.tool_prompt_type}"
    
    MAX_TURNS_STR = "" if args.max_turns == 5 else f"-turns-{args.max_turns}"
    NUM_MAJORITY_STR = "" if args.num_majority == 1 else f"-majority-{args.num_majority}"
    if MAJORITY_FINAL and NUM_MAJORITY > 1:
        NUM_MAJORITY_STR += "-final"

    DEFAULT_TEMP, DEFAULT_TOP_P = 0.6, 0.95
    TEMP_STR = "" if TEMP == DEFAULT_TEMP else f"-temp-{TEMP}"
    TOP_P_STR = "" if TOP_P == DEFAULT_TOP_P else f"-topp-{TOP_P}"
    HINT_STR = "" if args.apply_error_analysis else "-no-hint"
    if args.apply_hint_instructions:
        HINT_STR += "-with-hint-instructions"
        
    DEBUG_STR = "-debug" if args.debug else ""
    EARLY_STOP_STR = ""
    if args.max_turns > 1:
        EARLY_STOP_STR =  "-no-early-stop" if args.no_early_stop else ""
    
    if not args.error_analysis_file and args.apply_error_analysis:
        raise ValueError("Please provide --error_analysis_file to indicate the error analysis file used.")

    ERROR_ANALYSIS_STR = "" if (not args.apply_error_analysis or not args.error_analysis_file or "training_error_analysis_0911" in args.error_analysis_file) else f"-{os.path.basename(args.error_analysis_file).replace('.csv','').replace('.txt','')}" 

    if BACKEND_STR == "openai":
        model_name_str = f"openai-{OPENAI_MODEL}{MAX_TURNS_STR}{NUM_MAJORITY_STR}{TEMP_STR}{TOP_P_STR}{SYSTEM_PROMPT_TYPE_STR}{USER_PROMPT_TYPE_STR}{TOOL_PROMPT_TYPE_STR}{ERROR_ANALYSIS_STR}{EARLY_STOP_STR}{HINT_STR}{DEBUG_STR}"
    else:
        GPT_REASONING_STR = f"-reasoning-{args.reasoning_effort}" if args.gpt_oss else ""
        model_name_str = f"{MODEL_NAME}{GPT_REASONING_STR}{MAX_TURNS_STR}{NUM_MAJORITY_STR}{TEMP_STR}{TOP_P_STR}{SYSTEM_PROMPT_TYPE_STR}{USER_PROMPT_TYPE_STR}{TOOL_PROMPT_TYPE_STR}{ERROR_ANALYSIS_STR}{EARLY_STOP_STR}{HINT_STR}{DEBUG_STR}"

    backend_str = BACKEND_STR + (f"_{args.suffix}" if args.suffix else f"_{args.seed}" if args.seed else "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    OUT_DIR = os.path.join(args.output_dir, os.path.basename(DATA_PATH).replace(".parquet", "").replace(".csv", "").replace(".json", "").replace(".jsonl", ""), model_name_str, backend_str)
            
    print("Output dir", OUT_DIR)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_PATH = os.path.join(OUT_DIR, "multi_turn_results.jsonl")
    

    if not args.force and os.path.exists(OUT_PATH):
        print(f"Output file {OUT_PATH} already exists, exit.")
        exit(0)
    
    if args.gpt_oss:
        from openai_harmony import (
            HarmonyEncodingName,
            load_harmony_encoding,
        )
                
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        
    if "sglang" in BACKEND_SCHEDULE:
        engine = sgl.Engine(
            model_path=MODEL_LOAD_PATH,   
            tp_size=TP_SIZE,
            dp_size=DP_SIZE,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            random_seed=args.seed,
            max_running_requests=64,
        )
        if args.gpt_oss:
            stop_token_ids = encoding.stop_tokens_for_assistant_actions()
            stop_tokens = list(set([tokenizer.decode([tid]) for tid in stop_token_ids] + [tokenizer.pad_token]))
        else:
            stop_tokens = [tokenizer.eos_token, tokenizer.pad_token]
        print(">>> SGLANG stop tokens:", stop_tokens)
        sglang_sampling_params  = {"temperature": TEMP, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS, "stop": stop_tokens}


    print("="*45)
    print("BACKEND_SCHEDULE:", " | ".join([f"{b.upper()}" for b in BACKEND_SCHEDULE]))
    print("="*45)    
    
    try:
        asyncio.run(main())
    finally:
        if "sglang" in BACKEND_SCHEDULE:
            engine.shutdown()
    
    print("Done!")
