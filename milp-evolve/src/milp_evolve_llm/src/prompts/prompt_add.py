

import random
from src.utils import given_data_marker, new_data_marker, given_parameters_marker, new_parameters_marker, given_constraints_marker, new_constraints_marker
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_ADD

SYSTEM_MESSAGE_A = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your objective is to rewrite a given MILP formulation into a more complex version.
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle
Be creative, but keep in mind the created MILP must be reasonable and must be modeling real world optimization problems.
"""

PROMPT1_A = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP by adding complexity to a given MILP:

1. **Describe the Given MILP in Detail**:
    - Place the MILP within a specific real-world context.
    - Clearly outline the context, constraints, objectives, and variables, emphasizing its practical significance.

2. **Summarize and Formulate a New MILP**:
   - Discuss how to introduce new constraints, variables, data, or change the objective function retain the original MILP's structure but add complexity in the context of the real-world application.
   - Ensure new constraints or objectives are linear. Use tricks to linearize if necessary.

3. **Generate New MILP Code Step-by-Step as follows**: """ + PROMPT_SYNTAX_ADD.format(additional_prompt="reflect the real world context") + PROMPT_REQ + f"""
- Do not include `{given_constraints_marker}`, `{new_constraints_marker}`, `{given_data_marker}`, `{new_data_marker}`, `{given_parameters_marker}`, or `{new_parameters_marker}` in your final code.

Output Format:

### Description of the given MILP:
    <text to explain of the given MILP and its real-world application>

### Description of New Constraints, Variables, Objectives, Data, or Parameters:
    Names: <The full words of the names of constraint/variable/objective/data/parameter which starts with the capital letter selected from above.>
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