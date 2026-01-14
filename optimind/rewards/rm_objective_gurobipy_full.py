import torch
import os 
import re
import copy
import time
import subprocess
import random
import sys
import importlib
import traceback
import types
import uuid
import multiprocessing as mp
from io import StringIO
import contextlib
import tempfile

# import the current file's directory's utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # add parent directory to path

from utils import *
        
################## helper functions end ##################
def compute_score(data_source, solution_str, ground_truth, extra_info=None, precision=None):
    """
    label: the expected objective
    """
    
    code_block = extract_python_code_block(solution_str)
    solution_dict = run_full_code(code_block, solver="gurobipy")
    objective = solution_dict.get("objective_value", None)

    if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
        # if any is correct then correct
        accuracy = score = float(any([compute_acc(objective, gt, precision=precision) for gt in ground_truth]))
    else:
        accuracy = score = float(compute_acc(objective, ground_truth, precision=precision))

    no_execution_error = float("failed" not in solution_dict["output"].lower() and solution_dict["traceback_str"].strip() == "")
    have_objective = float(objective is not None)

    return {
        "score": score, 
        "accuracy": accuracy, 
        "objective": objective if objective is not None else 0.0,
        "no_execution_error": no_execution_error, 
        "have_objective": have_objective,
    }
    
if __name__ == '__main__':
    import os
    import argparse
    import pdb
    import pandas as pd
    
    solution_str = """
analysisWe need to formulate: decision variables = pounds of each grain (continuous, non-negative). Minimize cost = 41 x1 + 36 x2 + 96 x3. Subject to nutrient constraints: 20x1+30x2+70x3 >= 110 (A); 10x1+10x2+0x3 >= 18 (B); 50x1+30x2+0x3 >= 90 (C); 6x1+2.5x2+10x3 >= 14 (D). Non-negativity.\n\nSolve linear program. Compute optimal solution manually? Let's try solve quickly.\n\nWe can attempt to find feasible combination with minimal cost.\n\nWe could use python after code runs. But we need to produce code; after execution we will get solution. So we will write code accordingly.\n\nWe also need to output minimum cost in cents.\n\nWe will implement, solve, retrieve objective value.\n\nReturn result.\n\nWe'll wait for user to execute.assistantfinal**Mathematical formulation**\n\nLet  \n\n* \\(x_1\\) = pounds of Grain\u202f1 to use (continuous, \\(x_1 \\ge 0\\))  \n* \\(x_2\\) = pounds of Grain\u202f2 to use (continuous, \\(x_2 \\ge 0\\))  \n* \\(x_3\\) = pounds of Grain\u202f3 to use (continuous, \\(x_3 \\ge 0\\))\n\n**Objective (minimum cost in cents)**  \n\n\\[\n\\min \\; 41x_1 + 36x_2 + 96x_3\n\\]\n\n**Nutrient constraints**\n\n\\[\n\\begin{aligned}\n20x_1 + 30x_2 + 70x_3 &\\ge 110 \\qquad\\text{(Nutrient\u202fA)}\\\\\n10x_1 + 10x_2 + 0x_3 &\\ge 18 \\qquad\\text{(Nutrient\u202fB)}\\\\\n50x_1 + 30x_2 + 0x_3 &\\ge 90 \\qquad\\text{(Nutrient\u202fC)}\\\\\n 6x_1 + 2.5x_2 + 10x_3 &\\ge 14 \\qquad\\text{(Nutrient\u202fD)}\n\\end{aligned}\n\\]\n\n\\(x_1, x_2, x_3 \\ge 0\\)\n\n---\n\n**Gurobi implementation**\n\n```python\nimport gurobipy as gp\nfrom gurobipy import GRB\n\ndef solve_wool_feeding():\n    # create model\n    model = gp.Model(\"wool_feeding\")\n\n    # decision variables: pounds of each grain (continuous, non\u2011negative)\n    x1 = model.addVar(name=\"Grain1\", lb=0.0, vtype=GRB.CONTINUOUS)\n    x2 = model.addVar(name=\"Grain2\", lb=0.0, vtype=GRB.CONTINUOUS)\n    x3 = model.addVar(name=\"Grain3\", lb=0.0, vtype=GRB.CONTINUOUS)\n\n    # objective: minimize total cost (cents)\n    model.setObjective(41 * x1 + 36 * x2 + 96 * x3, GRB.MINIMIZE)\n\n    # nutrient constraints\n    model.addConstr(20 * x1 + 30 * x2 + 70 * x3 >= 110, name=\"Nutrient_A\")\n    model.addConstr(10 * x1 + 10 * x2               >= 18,  name=\"Nutrient_B\")\n    model.addConstr(50 * x1 + 30 * x2               >= 90,  name=\"Nutrient_C\")\n    model.addConstr( 6 * x1 + 2.5 * x2 + 10 * x3   >= 14,  name=\"Nutrient_D\")\n\n    # solve\n    model.optimize()\n\n    if model.status == GRB.OPTIMAL:\n        cost = model.objVal\n        print(f\"Optimal cost: {cost:.2f} cents\")\n        print(f\"Grain 1: {x1.X:.4f} lb\")\n        print(f\"Grain 2: {x2.X:.4f} lb\")\n        print(f\"Grain 3: {x3.X:.4f} lb\")\n        return cost\n    else:\n        print(\"No optimal solution found.\")\n        return None\n\nif __name__ == \"__main__\":\n    solve_wool_feeding()\n```\n\nRunning the script will display the minimum cost (in cents) together with the optimal amounts of each grain. The variable `model` is used as required, all variables are continuous, and constraints are expressed with `>=` (implemented via `>=` in Gurobi, which internally translates to the appropriate linear form)."
"""

    ground_truth = 2.0
    
    score = compute_score("math", solution_str, ground_truth)
    print("score:", score)
