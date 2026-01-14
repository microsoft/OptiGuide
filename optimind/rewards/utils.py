import re
import json
    

def extract_python_code_block(
    response: str, allow_overlapping: bool = True, return_status: bool = False, 
    return_none_if_failed: bool = False,
) -> str | None:
    """
    Return the last non-empty Python code block in *response*,
    scanning from the end to the beginning.

    Non-empty = contains at least one character other than whitespace or a dot.
    Supports three block syntaxes:
      1) ```python ... ``` but when matching this we do not want to match ```python ... ```python
         (the closing ``` should not be immediately followed by python)
      2) <python> ... </python>
      3) <code> ... </code>

    If no closed fence is found, falls back to the tail after the last ```python.
    """

    def non_empty(block: str) -> bool:
        return bool(re.search(r"[^\s\.]", block))
    
    def get_output(result, success: bool):
        if return_status:
            return result, success
        else:
            return result

    # --- patterns ---
    # Standard (non-overlapping)
    # NOTE: closing ``` must NOT be immediately followed by "python"
    rx_fence = re.compile(r"```python\s*(.*?)\s*```(?!python)", re.DOTALL)
    rx_py    = re.compile(r"<python>(.*?)</python>", re.DOTALL)
    rx_code  = re.compile(r"<code>(.*?)</code>", re.DOTALL)

    if not allow_overlapping:
        # Reverse order by consuming all matches, then iterate backwards.
        for block in reversed([m.group(1) for m in rx_fence.finditer(response)]):
            if non_empty(block):
                return get_output(block.strip(), True)
        
        for block in reversed([m.group(1) for m in rx_py.finditer(response)]):
            if non_empty(block):
                return get_output(block.strip(), True)
            
        for block in reversed([m.group(1) for m in rx_code.finditer(response)]):
            if non_empty(block):
                return get_output(block.strip(), True)
    else:
        # Overlapping via lookahead (captures the inner group only).
        # Same constraint: closing ``` must NOT be immediately followed by "python"
        rx_fence_ov = re.compile(r"(?s)(?=(```python\s*(.*?)\s*```(?!python)))")
        rx_py_ov    = re.compile(r"(?s)(?=(<python>(.*?)</python>))")
        rx_code_ov  = re.compile(r"(?s)(?=(<code>(.*?)</code>))")

        for block in reversed([m.group(2) for m in rx_fence_ov.finditer(response)]):
            if non_empty(block):
                return get_output(block.strip(), True)
        for block in reversed([m.group(2) for m in rx_py_ov.finditer(response)]):
            if non_empty(block):
                return get_output(block.strip(), True)
        for block in reversed([m.group(2) for m in rx_code_ov.finditer(response)]):
            if non_empty(block):
                return get_output(block.strip(), True)
    
    if return_none_if_failed:
        return get_output(None, False)
    
    # Fallback: unterminated ```python at the end — return tail if non-empty
    start = response.rfind("```python")
    if start != -1:
        tail = response[start + len("```python"):].strip()
        if non_empty(tail):
            return get_output(tail, False)

    # Nothing suitable found — return stripped whole response (original fallback)
    # This is also for cases where the response is a code (extracted from a code block before), so we just return it.
    return get_output(response.strip(), False)


def safe_eval(value):
    if isinstance(value, (int, float)):
        return round(value, 5)
    
    # First attempt: direct safe eval
    try:
        return round(eval(value), 5)
    except:
        # Second attempt: strip non-numeric/operator chars
        try:
            cleaned = re.sub(r"[^0-9eE\.\+\-\*/\(\)]", "", value)
            return round(eval(cleaned), 5)
        except:
            return None

def compute_acc(pred, true, precision=None):
    if precision is None: precision = 1e-6
    
    if pred is None or true is None:
        return float(pred is None and true is None)
    
    try:
        if precision > 1e-3:
            acc = abs(pred - true) < precision
        else:
            acc = abs(pred - true) / (abs(true) + 1) < precision
    except:
        acc = 0
    return acc
    

def run_full_code(code, solver="gurobipy", time_limit=10, tol=1e-7, return_code=False):
    IMPORTS = {
        "gurobipy": "import os\nimport gurobipy as gp\nfrom gurobipy import Model, GRB, quicksum, LinExpr, QuadExpr, Var, Constr",
    }

    ADD_SCRIPT_TOL = {
        "gurobipy": '{m}.setParam("IntFeasTol", {tol})',
    }  # Only support gurobipy right now

    # Use {m} placeholder; we'll format with detected model var name
    ADD_SCRIPT_FMT = {
        "gurobipy": 'if {m}.status == GRB.OPTIMAL:\n    print("Just print the best solution:", {m}.objVal)\nelse:\n    print("No Best Solution")',
    }

    if solver not in IMPORTS:
        raise ValueError(f"Unsupported solver: {solver}")

    if "```python" in code:
        maybe = extract_python_code_block(code)
        if maybe:
            code = maybe

    code_lines = code.strip().splitlines()

    # --- Detect solve/optimize line by solver (keep the last match) ---
    model_name = None
    solve_index = None
    solve_indent = 0

    pat = re.compile(r'([A-Za-z_]\w*)\s*\.\s*optimize\s*\(')
    # Now: use the *last* such call in the code.
    for i, line in enumerate(code_lines):
        m = pat.search(line)
        if m:
            model_name = m.group(1)
            solve_index = i
            solve_indent = len(line) - len(line.lstrip())

    if solve_index is None or model_name is None:
        ret = {
            "output": f"Failed: No detectable optimize call for solver '{solver}'.",
            "sol_output": f"Failed: No detectable optimize call for solver '{solver}'.",
            "objective_value": None,
            "traceback_str": "",
        }
        if return_code:
            ret["code"] = code
        return ret

    # Optional tol injection
    insert_idx_tol = solve_index
    if tol is not None and solver in ADD_SCRIPT_TOL and ADD_SCRIPT_TOL[solver]:
        tol_line = ADD_SCRIPT_TOL[solver].format(m=model_name, tol=tol)
        aligned_tol_line = (" " * solve_indent) + tol_line
        code_lines.insert(insert_idx_tol, aligned_tol_line)
        solve_index += 1  # Adjust since we inserted a line

    insert_idx = solve_index + 1

    # Align inserted script to the same indentation as the solve/optimize call
    template = ADD_SCRIPT_FMT[solver]
    add_script = template.format(m=model_name) if "{m}" in template else template
    aligned_add_script = [(" " * solve_indent) + l for l in add_script.splitlines()]
    new_code_lines = code_lines[:insert_idx] + aligned_add_script + code_lines[insert_idx:]

    program = f"{IMPORTS[solver]}\n\n" + "\n".join(new_code_lines)

    import tempfile, uuid, subprocess, sys
    try:
        import os  # may already be in IMPORTS; safe to import again
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as workdir:
        script_path = os.path.join(workdir, f"prog_{uuid.uuid4().hex}.py")
        with open(script_path, "w") as f:
            f.write(program)

        try:
            res = subprocess.run(
                [sys.executable, script_path],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=time_limit,
            )
            output = res.stdout
            traceback_str = res.stderr
        except subprocess.TimeoutExpired:
            err = "Failed: Execution timed out"
            return {"output": err, "sol_output": err,
                    "objective_value": None, "traceback_str": err}
        except Exception as e:
            err = f"Failed: {str(e)[:200]}"
            return {"output": err, "sol_output": err,
                    "objective_value": None, "traceback_str": err}

    # Extract numeric objective from the standardized print
    match = re.search(r"Just print the best solution:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    objective = float(match.group(1)) if match else None

    ret = {
        "output": output,
        "sol_output": output,
        "objective_value": objective,
        "traceback_str": traceback_str
    }
    if return_code:
        ret["code"] = program
    return ret


def compute_score_history_output(data_source, history_list, ground_truth, extra_info=None, precision=None):
    user_idxs = [i for i, turn in enumerate(history_list) if turn['role'] == 'user']
    objective = None
    for idx in reversed(user_idxs):
        output_str = history_list[idx]['content']
        potential_query = history_list[idx].get("query", "")
        if "No python code block found. Skipping code execution." in output_str or potential_query:
            continue
        else:
            match = re.search(r"Just print the best solution:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output_str)
            if match:
                objective = float(match.group(1))
            break
                
    if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
        # if any is correct then correct
        accuracy = score = float(any([compute_acc(objective, gt, precision=precision) for gt in ground_truth]))
    else:
        accuracy = score = float(compute_acc(objective, ground_truth, precision=precision))

    no_execution_error = float(objective is not None)
    have_objective = float(objective is not None)

    return {
        "score": score, 
        "accuracy": accuracy, 
        "objective": objective if objective is not None else 0.0,
        "no_execution_error": no_execution_error,   # this isnt accurate
        "have_objective": have_objective,
    }

# if __name__ == '__main__':
#     s = "```python\nimport gurobipy as gp\nfrom gurobipy import GRB\n\n# ------------------------------------------------------------------\n# Data (0\u2011based indices)\n# ------------------------------------------------------------------\nopen_cost = [\n    151000, 192000, 114000, 171000, 160000,\n    120000, 182000, 186000, 174000\n]\n\n# Shipping costs per unit: 9 centers \u00d7 5 stores\n#            Store1 Store2 Store3 Store4 Store5\nship_cost = [\n    [3, 5, 4, 3, 5],   # Center 1\n    [2, 4, 2, 4, 5],   # Center 2\n    [1, 4, 2, 5, 4],   # Center 3\n    [1, 1, 3, 3, 2],   # Center 4\n    [4, 4, 3, 4, 4],   # Center 5\n    [1, 3, 5, 3, 5],   # Center 6\n    [1, 2, 4, 1, 4],   # Center 7\n    [2, 2, 1, 2, 5],   # Center 8\n    [2, 4, 4, 4, 4],   # Center 9\n]\n\ndemand = [908, 434, 605, 480, 961]\n\ncapacity = [1825, 1821, 1365, 1929, 1502,\n            1201, 1529, 961, 1001]\n\n# ------------------------------------------------------------------\n# Model\n# ------------------------------------------------------------------\nmodel = gp.Model(\"GlobalSupply\")\n\ny = model.addVars(len(open_cost), vtype=GRB.BINARY, name=\"open\")\nx = model.addVars(len(demand), len(open_cost), vtype=GRB.CONTINUOUS, name=\"ship\")\n\n# Demand constraints\nfor i in range(len(demand)):\n    model.addConstr(\n        gp.quicksum(x[i, j] for j in range(len(open_cost))) == demand[i],\n        name=f\"demand_{i}\"\n    )\n\n# Capacity constraints\nfor j in range(len(open_cost)):\n    model.addConstr(\n        gp.quicksum(x[i, j] for i in range(len(demand))) <= capacity[j] * y[j],\n        name=f\"capacity_{j}\"\n    )\n\n# Objective\nobj = gp.quicksum(open_cost[j] * y[j] for j in range(len(open_cost))) + \\\n      gp.quicksum(ship_cost[j][i] * x[i, j]\n                  for i in range(len(demand))\n                  for j in range(len(open_cost)))\nmodel.setObjective(obj, GRB.MINIMIZE)\n\n# ------------------------------------------------------------------\n# Solve\n# ------------------------------------------------------------------\n# In this execution environment the method is named `solve`\nif hasattr(model, \"solve\"):\n    model.solve()\nelse:\n    model.optimize()\n\n# ------------------------------------------------------------------\n# Output\n# ------------------------------------------------------------------\nif model.status == GRB.OPTIMAL:\n    total_cost = model.objVal\n    print(f\"\\nOptimal total cost: ${total_cost:,.0f}\")\n\n    print(\"\\nCenters to open:\")\n    for j in range(len(open_cost)):\n        if y[j].X > 0.5:\n            print(f\"  Center {j+1}: open (${open_cost[j]:,.0f})\")\n\n    print(\"\\nShipping plan (units):\")\n    for i in range(len(demand)):\n        plan = []\n        for j in range(len(open_cost)):\n            if x[i, j].X > 1e-6:\n                plan.append(f\"C{j+1}:{x[i, j].X:.1f}\")\n        print(f\"Store {i+1:>2}: \" + \" -> \".join(plan))\nelse:\n    print(\"No optimal solution found.\")\n```"
#     code_block = extract_python_code_block(s)
#     print("Extracted code block:")
#     print(code_block)
#     ret = run_full_code(code_block)
#     print(ret)
