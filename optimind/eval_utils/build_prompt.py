import random

def build_prompt_sirl_system(solver="gurobipy"):
    return f"""You are a helpful Assistant with expertise in mathmetical modeling and the {solver} solver. When the User provides an OR question, you will analyze it, build a detailed mathematical model, and provide the {solver} code to solve it.

    Your response should follow these steps:
    1.  <think> 
Carefully analyze the problem to identify decision variables, objective, and constraints.
</think>
    2.  <model>Develop a complete mathematical model, explicitly defining:
        * Sets
        * Parameters
        * Decision Variables (and their types)
        * Objective Function
        * Constraints</model>
    3.  <python>Provide the corresponding {solver} Python code to implement the model.</python>

    The output must be in Markdown format, with each step enclosed in the specified tags.

The output must be in Markdown format, with each step enclosed in the specified tags.
"""

def build_prompt_sirl(question, solver="gurobipy"):
    return f"""Solve the following mathematical modeling problem:

{question}

Think step by step."""


instruction_list = [
    "Below is an operations research question. Build a mathematical model and corresponding Python code using `{solver}` to solve it.",
    "Analyze the given operations research problem, construct a mathematical model, and write Python code with {solver} to find the optimal solution.", 
    "Solve this optimization challenge by first writing out the complete mathematical model, then translating it into Python code using the {solver} package. Include comments explaining your implementation.", 
    "This is an operations research problem. Follow this structured approach: Begin with understanding the problem → Identify the key variables → Analyze the constraints → Develop the mathematical model → Solve it programmatically using {solver} in Python. Provide clear reasoning and explanations at each stage.",
    "Build an optimization model for this problem and solve it using {solver}. Your solution should include both the mathematical formulation and working Python code.",
    """Take the following steps to solve this optimization problem:
- Start with careful problem analysis
- Determine appropriate variable types
- Develop mathematical model with proper constraints
- Implement solution using {solver}
Provide clear explanations for your modeling decisions.""",
    """Let's solve this optimization challenge methodically:
1. First share your initial analysis of the problem
2. Then identify and classify all variables needed
3. Develop the complete mathematical formulation
4. Show how to implement in {solver}
Explain your reasoning throughout the process.""",
    "Using {solver}, write Python code to solve the following optimization problem. Start by developing the mathematical model before implementing the solution.",
    "We are presented with an operations research problem. Solve it step-by-step: 1) Identify and list the decision variables 2) Clearly define the objective 3) Outline all relevant constraints 4) Formulate the mathematical model 5) Implement and solve it using {solver} in Python, providing explanations at every step.",
    "Create a complete solution that includes: 1) Mathematical formulation 2) Python code using {solver} 3) Results interpretation. Ensure all variables and constraints are properly defined.",
    "Presented below is an operations research problem. Approach this systematically: What are we trying to optimize? What are the main constraints? After analyzing these elements, construct a mathematical model and solve it using {solver} in Python.",
    "Here is an operations research problem to solve. Tackle this problem carefully: Start by determining the goal we want to optimize, then pinpoint the constraints. Build the mathematical model and implement it using {solver} in Python, explaining each decision along the way.",
    "Please solve this optimization problem by first creating a mathematical formulation, then implement the solution using {solver} in Python.",
    """Let's solve this step-by-step:
1) Analyze the problem requirements carefully
2) Identify decision variables and their domains
3) Write out the complete mathematical model
4) Implement solution in {solver}
Remember to explain your thought process.""",
    "Please provide a step-by-step solution that includes the mathematical formulation and corresponding Python implementation using {solver}. Ensure proper documentation of your approach."
]


#### different solver import languages
solver_import_dict = {
    "gurobipy": "import gurobipy as gp\nfrom gurobipy import Model, GRB, quicksum, LinExpr, QuadExpr, Var, Constr",
}

solver_extra_info_dict = {}


input_format = """# Question: {question}

# Note:
- The Code must include:```python
{solver_import}
```
{solver_extra_info}
- Make sure the model variable is named `model`.
- Avoid using "<" and ">" in {solver} constraints; instead, use "<=" or ">=".
- Carefully determine whether the variable is an integer or continuous.
"""

def get_solver_import(solver: str) -> str:
    solver = "gurobipy" if solver not in solver_import_dict else solver
    return solver_import_dict.get(solver, "")

def get_solver_extra_info(solver: str) -> str:
    solver = "gurobipy" if solver not in solver_extra_info_dict else solver
    extra_info = solver_extra_info_dict.get(solver, "")
    return "- " + extra_info if extra_info != "" else ""
        
def build_prompt_optmath(question: str, instruction: str = "", solver="gurobipy") -> str:
    if instruction == "":
        instruction = random.choice(instruction_list).strip().format(solver=solver)
    
    solver_import = get_solver_import(solver)
    solver_extra_info = get_solver_extra_info(solver)
    input_format_str = input_format.format(question=question.strip(),solver_import=solver_import, solver=solver, solver_extra_info=solver_extra_info)
    return f"{instruction}\n{input_format_str}\n{question.strip()}\nThink step by step."
