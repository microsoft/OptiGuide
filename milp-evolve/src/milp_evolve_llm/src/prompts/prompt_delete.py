from src.prompts.prompt_common import PROMPT_REQ

SYSTEM_MESSAGE_D = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your task is to generate a new MILP by removing less important constraints, variables, or objectives from the given MILP as the given MILP is too big.
Be creative, but ensure the new MILP is realistic and models genuine optimization challenges.
"""


PROMPT1_D = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP by removing less important components from the given MILP code:

1. Provide a Detailed Description of the given MILP code.

2. Provide a Detailed Description of the new MILP code that removes the less important and components.
    - Explain how the new MILP removes the less important components from the given MILP.
    - You may modify the constraints, variables, or objectives, data generation or parameters to make the resulting MILP coherent.
    - The resulting MILP should still be complex and challenging to solve.

3. Generate the new MILP code by following these procedures step-by-step:
    - Modify Realistic Constraints, Variables, or Objectives:
        - Modify the `solve` function by removing less important constraints or variables, or simplifying the objectives.

    - Modify the Data Generation Procedure:
        - Modify the `get_instance` function by updating the res dictionary to support the modified optimization modeling.
        - Use functions like `random.rand` or `random.randint` or `np.random.normal` or `np.random.gamma` or `nx.erdos_renyi_graph` or `nx.barabasi_albert_graph` or similar functions to create datasets of varying sizes.
        - Ensure the modified data support the modified constraints, variables, or objectives.

    - Modify the Parameters:
        - Modify the parameters within the `if __name__ == '__main__'` block to support the data generation.
        - The value of each parameter should be a constant (e.g. integer, float, boolean, string). That is, if there is a tuple value, you should break down the tuple into individual parameters with constant values. If it is a more complicated data structure (a list, dictionary, set, or a function), please put the data structure in the `get_instance` function and only put the required constants to construct the data structure in the parameters dictionary.
        - Ensure parameters can be modified easily to scale up the problem or to alter its complexity.

""" + PROMPT_REQ + f"""

Output Format:

### Description of the given MILP:
    <text to describe in the given MILP>

### Description of the new MILP:
    <text to explain what need to be changed to remove the less important components from the given MILP>

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```

Here is the given MILP Code:
""" + "{code}"
