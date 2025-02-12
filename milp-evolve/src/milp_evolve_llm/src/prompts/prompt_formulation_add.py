

from src.utils import given_data_marker, new_data_marker, given_parameters_marker, new_parameters_marker, given_constraints_marker, new_constraints_marker
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_ADD

SYSTEM_MESSAGE_FA = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your objective is to choose a MILP formulation method to rewrite a given MILP formulation.
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
Be creative, but keep in mind the created MILP must be reasonable and must be modeling real world optimization problems.
"""

PROMPT1_FA = """
Follow these step-by-step instructions to generate a diverse and realistic MILP by adding complexity to a given MILP using a specific MILP formulation method:

1. **Describe the Given MILP in Detail**:
    - Place the MILP within a specific real-world context.
    - Clearly outline the context, constraints, objectives, and variables, emphasizing its practical significance.

2. **Choose a MILP Formulation Method** from the following options: """ + "{formulation_methods}" + f""" 
    - Explain how this method applies to the given MILP in a real-world context.
    - Discuss the relevance of the chosen method to enhancing the complexity and realism of the MILP formulation.

3. **Summarize and Formulate a New MILP**:
   - Retain the original MILP's structure but add complexity by incorporating the specific MILP formulation method .
   - Discuss how to introduce new constraints, variables, data, or change the objective function in the context of the real-world application.
   - Ensure new constraints or objectives are linear. Use tricks to linearize if necessary.

4. **Generate New MILP Code Step-by-Step as follows**: """ + PROMPT_SYNTAX_ADD.format(additional_prompt="reflect the given formulation method") + PROMPT_REQ + f"""
- Do not include `{given_constraints_marker}`, `{new_constraints_marker}`, `{given_data_marker}`, `{new_data_marker}`, `{given_parameters_marker}`, or `{new_parameters_marker}` in your final code.

Output Format:

### Description of the given MILP:
    <text to explain of the given MILP and its real-world application>

### Description of the chosen MILP formulation method and its application to the given MILP:
    <text to explain the chosen MILP formulation method, its relevance to the given MILP in the context of the real-world application>

### Description of New Constraints, Variables, Objectives, Data, or Parameters:
    <text to describe how to incorporate the specific formulation method to the given MILP by proposing new constraints, variables, objectives, data, or parameters>
    <New [constraint/variable/objective/data/parameter] 1: description,
     New [constraint/variable/objective/data/parameter] 2: description,
     ...
    >

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```
    
Here is the given MILP Code:
""" + "{code}"