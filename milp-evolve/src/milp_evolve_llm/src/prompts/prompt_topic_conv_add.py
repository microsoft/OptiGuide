

from src.utils import given_data_marker, new_data_marker, given_parameters_marker, new_parameters_marker, given_constraints_marker, new_constraints_marker
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_ADD, letter2word

SYSTEM_MESSAGE_CONVA = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your objective is to rewrite a given MILP formulation into a more complex version.
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle
Be creative, but keep in mind the created MILP must be reasonable and must be modeling real world optimization problems.
"""

PROMPT1_CONVA = """
Follow these step-by-step instructions to generate a diverse and realistic MILP by adding complexity to a given MILP from an insightful conversation:

1. **Create a Conversation**:
    - Generate a dialogue between an expert (assistant) and a novice (user) in the mixed integer linear programming (MILP) domain.
    - Conversation Topic: The dialogue should be about """ + "{topic}" + f""".
    - Focus: The user inquires about adding complexity to a given MILP to better model a real-world problem.
    - Requirements:
        - Discuss topics like objective value, data, constraints, or problem formulation.
        - Ensure the conversation is theoretical, mathematical, and coding-oriented (Python PySCIPOPT).
        - The expert should explain simply, and the novice should ask clarifying questions.
    - Keywords: Include terms starting with """ + "{capital_letters}" + f""" related to the task.

2. **Summarize and Formulate a New MILP**:
    - Retain the original MILP's structure but add complexity by summarizing the conversation above around the specific topic.
    - Discuss how to introduce new constraints, variables, data, or change the objective function.
    - Ensure new constraints or objectives are linear. Use tricks to linearize if necessary.

3. **Generate New MILP Code Step-by-Step as follows**: """ + PROMPT_SYNTAX_ADD.format(additional_prompt="reflect the conversation around the given topic") + PROMPT_REQ + f"""
- Do not include placeholder comments like `{given_constraints_marker}`, `{new_constraints_marker}`, `{given_data_marker}`, `{new_data_marker}`, `{given_parameters_marker}`, or `{new_parameters_marker}` in the final code.
""" + """
Output Format:

### Two people conversation around the given topic:
    <KEYWORD: your keywords with first letters {capital_letters2} here.>
    <CONVERSATION related to KEYWORD: 
     User: xxx
     Assistant: yyy 
     User: xxx
     Assistant: yyy 
     ...
     >

### Description of New Constraints, Variables, Objectives, Data, or Parameters:
    <text to describe how to distill the conversation above to the given MILP by proposing new constraints, variables, objectives, data, or parameters>
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