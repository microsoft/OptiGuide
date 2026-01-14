import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math
import ast
import pandas as pd

import sys

# file's parent's parent directory
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
from rewards.utils import extract_python_code_block as extract_python_code_block_helper


# ─────────────────── Helper: load indented JSON objects ───────────────────────────────
def load_indented_json_objects(file_path: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    buffer: List[str] = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            buffer.append(line)
            try:
                obj = json.loads("".join(buffer))
                objects.append(obj)
                buffer = []  # reset for next object
            except json.JSONDecodeError:
                # incomplete object, keep accumulating
                continue
    return objects


# ─────────────── Helper: resolve local snapshot from HF cache ───────────────
def resolve_model_path(
    model_path: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> str:
    """
    If model_path is a local dir, return it.
    Otherwise, return a *local snapshot path* for our private repos
    (using snapshot_download). For public HF models, return the
    original model_path so that transformers/vLLM can handle it.
    """
    p = Path(model_path).expanduser()
    if p.exists():
        return str(p)


    # If we are offline and don't have a specific revision, try to find the 
    # most recent snapshot in the Hugging Face cache structure.
    if local_files_only and not revision:
        try:
            if cache_dir:
                base_cache = Path(cache_dir)
            else:
                base_cache = Path(os.getenv("HF_HOME", "~/.cache/huggingface/hub")).expanduser()
            
            repo_dir_name = f"models--{model_path.replace('/', '--')}"
            snapshot_parent = base_cache / repo_dir_name / "snapshots"

            if snapshot_parent.exists():
                snapshots = [x for x in snapshot_parent.iterdir() if x.is_dir()]
                
                if snapshots:
                    # Sort by modification time, newest first
                    snapshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    # Set revision to the most recent folder name (the hash)
                    revision = snapshots[0].name
                    print(f"[_resolve_model_path] Offline mode: Auto-detected latest snapshot revision: {revision}")
        except Exception as e:
            # We do not want this logic to crash the script; strictly fallback to standard behavior
            print(f"[_resolve_model_path] Warning: Attempted to find cached snapshot but failed: {e}")

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as e:
        if local_files_only:
            raise FileNotFoundError(
                f"Offline mode: model '{model_path}' not found in local cache. "
                f"Run once without --offline_only to download. Original error:\n{e}"
            )
        raise

    return model_path


def setup_hf_env(args) -> None:
    """Configure HF cache / offline flags from CLI args."""
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = os.path.expanduser(args.hf_cache_dir)

    if args.offline_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def setup_model_and_env(args):
    """
    Compute derived global config that depends on args and environment.
    Returns a dict with MODEL_PATH, MODEL_NAME, MODEL_LOAD_PATH, etc.
    """
    setup_hf_env(args)

    model_path = args.model_path.rstrip("/")

    if model_path.startswith("/mnt/ddn"):
        pieces = model_path.split("/")
        if "actor_hf" in model_path:
            name_parts = pieces[-3:-1]
        else:
            name_parts = pieces[-2:]
        model_name = "-".join([p.lower() for p in name_parts])
    else:
        model_name = os.path.basename(model_path).lower()

    if "lmsys" in model_path:
        model_name = "lmsys-" + model_name

    if args.backend_str == "azure":
        model_load_path = model_path
    else:
        model_load_path = resolve_model_path(
            model_path,
            cache_dir=os.path.expanduser(args.hf_cache_dir) if args.hf_cache_dir else None,
            revision=args.hf_revision,
            local_files_only=args.offline_only,
        )
        print(f"[HF load] Using path: {model_load_path} (offline={args.offline_only})")

    return {
        "MODEL_PATH": model_path,
        "MODEL_NAME": model_name,
        "MODEL_LOAD_PATH": model_load_path,
    }


################ utils for error analysis ################ 

def as_list(obj) -> List[str]:
    """Parse a list-like field that may be a string repr, list, or NaN."""
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    s = str(obj).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
        return [s]
    except Exception:
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]

def parse_list_of_pairs(obj: Any) -> List[Tuple[str, str]]:
    """
    Expect a list of (mistake, hint) pairs. Robust to strings via literal_eval.
    """
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return []
    if isinstance(obj, list):
        out: List[Tuple[str, str]] = []
        for x in obj:
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append((str(x[0]), str(x[1])))
            else:
                out.append((str(x), ""))
        return out 
    s = str(obj).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        return parse_list_of_pairs(val)
    except Exception:
        pairs = manual_parse_list_of_pairs(s)
        return pairs
    
def manual_parse_list_of_pairs(s):
    pairs: List[Tuple[str, str]] = []
    current_pair = []
    for line in s.splitlines():
        line = line.strip().rstrip(",")
        if line: 
            if line.startswith("(") and line.endswith(")"):
                # eval this line
                try:
                    tup = ast.literal_eval(line)
                    if isinstance(tup, (list, tuple)) and len(tup) >= 2:
                        pairs.append((str(tup[0]), str(tup[1])))
                except:
                    pairs.append((line, ""))
            elif line.startswith("("):
                # print("Found start")
                current_pair = [line.lstrip("(")]
            elif  line.endswith(")") and current_pair:
                # print("Found end", line)
                current_pair.append(line.rstrip(")"))
                if len(current_pair) >= 2:
                    pairs.append((current_pair[0], current_pair[1]))
                current_pair = []   
    pairs = [(a.strip().strip('"').strip("'"), b.strip().strip('"').strip("'")) for a, b in pairs]

    return pairs

def load_ea_pairs(ea, args):
    # define once (removed duplicate)
    class_to_pairs: Dict[str, List[Tuple[str, str]]] = {}

    if "problem_class" not in ea.columns or "error analysis" not in ea.columns:
        raise ValueError("error_analysis_file must have columns 'problem_class' and 'error analysis'.")

    for _, row in ea.iterrows():
        classes = as_list(row["problem_class"])

        # format error analysis in case it's just a single tuple instead of a list
        error_analysis_str = row["error analysis"]
        if not isinstance(error_analysis_str, str):
            # cannot find a valid error analysis for this
            continue

        error_analysis_str = error_analysis_str.strip()
        if not error_analysis_str.startswith("["):
            error_analysis_str = "[" + error_analysis_str
        if not error_analysis_str.endswith("]"):
            error_analysis_str = error_analysis_str + "]"

        pairs_list = parse_list_of_pairs(error_analysis_str)
        if not classes or not pairs_list:
            continue
        for c in classes:
            class_to_pairs.setdefault(c, []).extend(pairs_list)

    # ----- move dedup BEFORE building all_hints_block -----
    for c, pairs in list(class_to_pairs.items()):
        seen = set()
        dedup: List[Tuple[str, str]] = []  # <— type corrected
        for a, b in pairs:
            key = (a.strip(), b.strip())
            if key not in seen:
                seen.add(key)
                dedup.append(key)
        class_to_pairs[c] = dedup
    return class_to_pairs

# hint instruction
def get_hint_instructions():
    lines = []
    lines.append(
        "Instructions for applying error-analysis hints:\n"
        "- Review the provided hints and identify which ones are applicable to this problem.\n"
        "- Please apply ALL applicable hints.\n"
        "- Before applying any hint or writing constraints, check the sign and direction of every variable and coefficient for consistency (e.g., profit = revenue - cost; capacities as ≤; flow conservation as =).\n"
        "\n"
        "General modeling checklist (follow rigorously):\n"
        "- Units: Use correct units everywhere and ensure the objective’s units match the goal (e.g., dollars for cost/profit, distance for TSP, time for scheduling). Do not mix units (e.g., minutes with hours, dollars with 1000 dollars) without converting.\n"
    )
    return "".join(lines)

def _format_error_pair(error_pair) -> str:
    """
    Normalize one error-analysis pair into a single string.
    Accepts tuples/lists (mistake, hint) or a bare string.
    Tries to recover if one side is empty by literal_eval of the other.
    """
    # Case 1: pair-like (tuple/list)
    if isinstance(error_pair, (list, tuple)):
        # make sure we're working with a tuple, then pad with empty strings
        a, b = (tuple(error_pair) + ("", "",))[:2]   # <-- FIX: tuple(...) + ("","")
        # If one side is blank, try to parse the other as a tuple-like string
        if isinstance(a, str) and not a.strip():
            try:
                parsed = ast.literal_eval(b)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    a, b = parsed[0], parsed[1]
            except Exception:
                pass
        if isinstance(b, str) and not b.strip():
            try:
                parsed = ast.literal_eval(a)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    a, b = parsed[0], parsed[1]
            except Exception:
                pass

        a = (a if isinstance(a, str) else str(a)).strip()
        b = (b if isinstance(b, str) else str(b)).strip()

        if a and b:
            return f"{a}, {b}"
        return a or b or ""

    # Case 2: bare string (or anything else)
    s = str(error_pair).strip()
    return s


def build_error_analysis_str(error_pairs):
    lines = []
    if len(error_pairs) > 0:
        lines.append("\nBelow are hints for avoiding common mistakes often seen for this problem type. Avoid them if applicable.\n")
    for k, error_pair in enumerate(error_pairs):
        body = _format_error_pair(error_pair)
        error_analysis_str = f"Error analysis #{k}: {body}"
        lines.append(error_analysis_str + "\n")
    return "".join(lines)

