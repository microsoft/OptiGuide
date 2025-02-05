from src.prompts.prompt_common import PROMPT_REQ

SYSTEM_MESSAGE_RM = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your task is to generate a new MILP by removing redundant constraints, variables, or objectives from the given MILP and replacing them with more diverse and less redundant counterparts.
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
Be creative, but ensure the new MILP is realistic and models genuine optimization challenges.
"""


PROMPT1_RM = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP by removing redundancy and introducing novel components to the given MILP code:

1. Provide a Detailed Description of the given MILP code.
    - Identify the redundancy in the constraints, variables, or objectives.
    - Examples of redundancy include constraints of similar types, unused variables or parameters, duplicate data generation procedure, or objectives that can be simplified.

2. Provide a Detailed Description of the new MILP code that removes the redundancy and introduces novel, more diverse components.
    - Explain how the new MILP removes redundancies and introduces novel, more diverse components.
    - Examples include introducing different types of MILP constraints, various variable types, or more complex objectives reflecting diverse optimization goals.

3. Generate the new MILP code by following these procedures step-by-step:
    - Generate Realistic Constraints, Variables, or Objectives:
        - Inside the `solve` function, define the constraints, variables, or objective to reflect the proposed modifications for a more diverse optimization problem.
        - Ensure the new optimization problem challenges the solver and varies in difficulty, contributing to diverse solving times and gap improvements.

    - Generate the Data Generation Procedure:
        - Inside the `get_instance` function, define the data generation and the res dictionary to generate more diverse data.
        - Besides functions like `random.rand`, `random.randint`, `np.random.normal`, `np.random.gamma`, `nx.erdos_renyi_graph`, `nx.barabasi_albert_graph`, use additional functions with proper import statement to improve the date diversity.
        - Ensure the new generated data supports the modified constraints, variables, or objectives.

    - Generate the Parameter Definitions:
        - Under the `if __name__ == '__main__'` block, generate the parameters to support the modified optimization problem and data generation procedure.
        - The value of each parameter should be a constant (e.g. integer, float, boolean, string). That is, if there is a tuple value, you should break down the tuple into individual parameters with constant values. If it is a more complicated data structure (a list, dictionary, set, or a function), please put the data structure in the `get_instance` function and only put the required constants to construct the data structure in the parameters dictionary.
        - Ensure parameters can be modified easily to scale up the problem or to alter its complexity.

""" + PROMPT_REQ + f"""

Output Format:

### Description of the redundancies in the given MILP:
    <text to explain the redundant components in the given MILP>

### Description of the new MILP:
    <text to explain the modifications in the new MILP to make it less redundant with more diverse components>

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```

Here is the given MILP Code:
""" + "{code}"
