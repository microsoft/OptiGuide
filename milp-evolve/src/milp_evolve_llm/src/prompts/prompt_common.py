from src.utils import given_data_marker, new_data_marker, given_parameters_marker, new_parameters_marker, given_constraints_marker, new_constraints_marker


PROMPT_SYNTAX_ADD = """
    - Add Constraints, Variables, or Objectives:
        - Introduce new elements to increase problem complexity while ensuring feasibility.
        - Place these in the `solve` function between `""" + given_constraints_marker + """` and `""" + new_constraints_marker + """` to {additional_prompt}.
    - Modify Data Generation:
        - Use functions like `random.rand` or `random.randint` or `np.random.normal` or `np.random.gamma` or `nx.erdos_renyi_graph` or `nx.barabasi_albert_graph` or similar functions to create diverse datasets.
        - Insert new data in the `get_instance` function between `""" + given_data_marker + """` and `""" + new_data_marker + """` to {additional_prompt}.
    - Update the Parameters:
        - Define new parameters in `if __name__ == '__main__'` between `""" + given_parameters_marker + """` and `""" + new_parameters_marker + """` to {additional_prompt}.
        - The value of each parameter should be a constant (e.g. integer, float, boolean, string). That is, if there is a tuple value, you should break down the tuple into individual parameters with constant values. If it is a more complicated data structure (a list, dictionary, set, or a function), please put the data structure in the `get_instance` function and only put the required constants to construct the data structure in the parameters dictionary.
        - Ensure parameters are adjustable to scale the problem's complexity.

"""

PROMPT_SYNTAX_MUTATE = """
    - Modify Realistic Constraints, Variables, or Objectives:
        - Modify the `solve` function by updating the constraints, variables, or objective to {prompt_optim}.
        - Ensure the modifications challenge the solver and vary in difficulty, contributing to diverse solving times and gap improvements.

    - Modify the Data Generation Procedure:
        - Modify the `get_instance` function by updating the res dictionary to {prompt_data}.
        - Use functions like `random.rand` or `random.randint` or `np.random.normal` or `np.random.gamma` or `nx.erdos_renyi_graph` or `nx.barabasi_albert_graph` or similar functions to create datasets of varying sizes.
        - Ensure the modified data support the modified constraints, variables, or objectives.

    - Modify the Parameters:
        - Modify the parameters within the `if __name__ == '__main__'` block to {prompt_param}.
        - The value of each parameter should be a constant (e.g. integer, float, boolean, string). That is, if there is a tuple value, you should break down the tuple into individual parameters with constant values. If it is a more complicated data structure (a list, dictionary, set, or a function), please put the data structure in the `get_instance` function and only put the required constants to construct the data structure in the parameters dictionary.
        - Ensure parameters can be modified easily to scale up the problem or to alter its complexity.

"""

PROMPT_SYNTAX_NEW = """
    - Generate Realistic Constraints, Variables, or Objectives:
        - Inside the `solve` function, define the constraints, variables, or objective to {prompt_optim}.
        - Ensure the new optimization problem challenge the solver and vary in difficulty, contributing to diverse solving times and gap improvements.

    - Generate the Data Generation Procedure:
        - Inside the `get_instance` function, define the data generation and the res dictionary to {prompt_data}.
        - Use functions like `random.rand` or `random.randint` or `np.random.normal` or `np.random.gamma` or `nx.erdos_renyi_graph` or `nx.barabasi_albert_graph` or similar functions to create datasets of varying sizes.
        - Ensure the new generated data support the generated constraints, variables, or objectives.

    - Generate the Parameters:
        - Under the `if __name__ == '__main__'` block, generate the parameters to {prompt_param}.
        - The value of each parameter should be a constant (e.g. integer, float, boolean, string). That is, if there is a tuple value, you should break down the tuple into individual parameters with constant values. If it is a more complicated data structure (a list, dictionary, set, or a function), please put the data structure in the `get_instance` function and only put the required constants to construct the data structure in the parameters dictionary.
        - Ensure parameters can be modified easily to scale up the problem or to alter its complexity.

"""


PROMPT_REQ = """
Ensure the Following:
- Code completeness:
    - Provide the COMPLETE, EXECUTABLE code, including all necessary library imports, the MILP class, and the parameters and code to call the MILP class. 
    - DO NOT OMIT ANY PART OF THE PROVIDED MILP CODE WITH `... (same as before) ...`, EVEN IF IT IS NOT MODIFIED. DO NOT INHERIT FROM A PREVIOUSLY DEFINED CLASS; INSTEAD, PROVIDE THE ENTIRE CODEBASE.
- Novelty and Increased difficulty for the new MILP while maintaining feasibility and reducing redundacy:
    - The new constraints, variables, objectives, data, and parameters types should be diverse, such as having different constraint types, and creative data generation schemes with correct syntax. 
    - The complexity of the new MILP can be achieved by more complicated constraint, objective, data generation scheme or larger parameter values.
    - Avoid adding redundant constraints, variables, or objectives that do not contribute to the problem's complexity, but do provide a clear, concise, and complete executable MILP code.
- Modularity and executability of the new MILP code:
    - Maintain function integrity by ensuring that no references to out-of-scope variables are used.
    - Define any new helper functions within the `get_instance` or `solve` functions, ensuring they are correctly scoped and called.
    - Use clear, descriptive names for new parameters and variables, ensuring they align with the real-world context and adhere to correct syntax.
    - Ensure the code remains modular and can easily scale to larger or different MILP problems by simply adjusting parameters, without altering core functions.
- Variable and Constraint Naming Convention:
    - Do not use parathesis or space in the variable and constraint names for the correctness of saved MILP instance files for the MILP solver. 
    - For example, if you want to name an edge variable for edge (u, v), you should name it as edge_u-v or edge_u_v.
"""

letter2word = {
    "A": "apple",
    "B": "banana",
    "C": "cherry",
    "D": "date",
    "E": "elderberry",
    "F": "fig",
    "G": "grape",
    "H": "honeydew",
    "I": "ice",
    "J": "jackfruit",
    "K": "kiwi",
    "L": "lemon",
    "M": "mango",
    "N": "nectarine",
    "O": "orange",
    "P": "papaya",
    "Q": "quince",
    "R": "raspberry",
    "S": "strawberry",
    "T": "tomato",
    "U": "ugli fruit",
    "V": "vanilla",
    "W": "watermelon",
    "X": "xray",
    "Y": "yolk",
    "Z": "zucchini",
}
