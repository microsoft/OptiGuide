

from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_NEW

SYSTEM_MESSAGE_TN = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your task is to generate a new MILP in the context of a specific topic that follows similar python syntax and structure as the given MILP but models a completely different optimization problem with a different real-world application.
This new MILP must be novel and feature different constraints, variables, objectives, data, and parameters than the given MILP. 
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle
Be creative, but ensure the new MILP is realistic and models genuine optimization challenges.
"""


PROMPT1_TN = """
Follow these step-by-step instructions to generate a diverse and realistic MILP in the context of a specific topic following the given MILP code structure:

1. Provide a Detailed Description of the python coding style of the given MILP code:
    - The description should only focus on the code style, but not the MILP details.

2. Provide a Detailed Description of new MILP code to suite a different optimization problem under the topic """ + "{topic}" + f""" with a specific real world application.
    - Discuss the novelty of the new MILP code in terms of constraints, variables, objective function, data and parameters and how they align with the given topic and the associated real-world scenario.

3. Generate the new MILP code by following the coding style of the given MILP code. For example, it must contain the following components"""+ PROMPT_SYNTAX_NEW.format(
                                    prompt_optim="reflect the given topic and the associated real world application",
                                    prompt_data="support the new constraints, variables or objective", prompt_param="support the data generation for the optimization") + PROMPT_REQ + f"""
- Avoid Repeating Existing Code. Make the modified constraints, variables, objectives, data, and parameters distinct from the original.

Output Format:

### Description of the python coding style of the given MILP:
    <text to explain of the python coding style>

### Description of the new MILP:
    <text to describe how to incorporate the specific topic to generate the new MILP>

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```
    
Here is the given MILP Code:
""" + "{code}"