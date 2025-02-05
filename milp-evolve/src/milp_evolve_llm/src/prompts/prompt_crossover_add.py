

from src.utils import given_data_marker, new_data_marker, given_parameters_marker, new_parameters_marker, given_constraints_marker, new_constraints_marker
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_ADD

SYSTEM_MESSAGE_CA = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your objective is to rewrite a given MILP formulation by incorporating iinformation (constraints, variables, objective, data, or parameters) from another MILP formulation.
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle
Be creative, but keep in mind the created MILP must be reasonable and must be modeling real world optimization problems.
"""

PROMPT1_CA = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP by incorporating information from another MILP to add complexity to a given MILP:

1. Provide a Detailed Description of the given MILP code:
    - Embed the MILP within a specific real-world application. 
    - Clearly describe the context, constraints, objective, and variables in a way that highlights its practical importance.

2. Provide a Detailed Description of the second MILP code that you should incorporate into the given MILP code:
    - Embed the MILP within the same specific real-world application as the given MILP code. 
    - Clearly describe the context, constraints, objective, and variables in a way that highlights its practical importance.
    - Explain the similarities and differences between the given MILP and the second MILP.

3. Explain the Modifications to the given MILP code and Their Impact:
    - Discuss how incorporating constraints, variables, or changing the objective function from the second MILP code to the given MILP code in the given real-world scenario.
    - The new MILP should retain the majority of the given MILP's structure, but make significant addition based on the second MILP codebase to make it more complex and challenging.
    - Ensure new constraints or objectives are linear. Use tricks to linearize if necessary.

4. **Generate New MILP Code Step-by-Step as follows**: """ + PROMPT_SYNTAX_ADD.format(additional_prompt="reflect a combination of both MILP code") + PROMPT_REQ + f"""
- Do not include `{given_constraints_marker}`, `{new_constraints_marker}`, `{given_data_marker}`, `{new_data_marker}`, `{given_parameters_marker}`, or `{new_parameters_marker}` in your final code.

Output Format:

### Description of the given MILP:
    <text to explain of the given MILP and its real-world application>

### Description of the second MILP:
    <text to explain of the second MILP in terms of the similarities and differences between the given MILP and the second MILP in a real-world application context>

### Description of New Constraints, Variables, Objectives, Data, or Parameters:
    <text to describe how to incorporate the second MILP to the given MILP by proposing new constraints, variables, objectives, data, or parameters>
    <New [constraint/variable/objective/data/parameter] 1: description,
     New [constraint/variable/objective/data/parameter] 2: description,
     ...
    >

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```

""" + "Here is the given MILP Code: {code}\n" + "Here is the second MILP code: {code2}"