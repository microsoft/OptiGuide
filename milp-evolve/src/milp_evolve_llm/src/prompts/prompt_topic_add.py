

from src.utils import given_data_marker, new_data_marker, given_parameters_marker, new_parameters_marker, given_constraints_marker, new_constraints_marker
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_ADD

SYSTEM_MESSAGE_TA = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your objective is rewrite a given MILP formulation in the context of a specific topic.
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle
Be creative, but keep in mind the created MILP must be reasonable and must be modeling real world optimization problems.
"""

PROMPT1_TA = """
Follow these step-by-step instructions to generate a diverse and realistic MILP by adding complexity to a given MILP in the context of a specific topic:

1. **Describe the Given MILP in Detail**:
    - Place the MILP within a specific real-world context under the topic """ + "`{topic}`" + f""".
    - Clearly outline the context, constraints, objectives, and variables, emphasizing its practical significance.

2. **Summarize and Formulate a New MILP**:
    - Retain the original MILP's structure but add complexity by placing the MILP the specific topic.
    - Ensure new constraints or objectives are linear. Use tricks to linearize if necessary.

. **Generate New MILP Code Step-by-Step as follows**: """ + PROMPT_SYNTAX_ADD.format(additional_prompt="reflect the given topic") + PROMPT_REQ + f"""
- Do not include `{given_constraints_marker}`, `{new_constraints_marker}`, `{given_data_marker}`, `{new_data_marker}`, `{given_parameters_marker}`, or `{new_parameters_marker}` in your final code.

Output Format:

### Description of the given MILP in the context of the given topic:
    <text to explain of the given MILP and its real-world application>

### Description of New Constraints, Variables, Objectives, Data, or Parameters:
    <text to describe how to incorporate the specific topic to the given MILP by proposing new constraints, variables, objectives, data, or parameters>
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
