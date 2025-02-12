

import random
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_NEW

SYSTEM_MESSAGE_N = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your task is to generate a new MILP that follows similar python syntax and structure as the given MILP but models a completely different optimization problem with a different real-world application. 
This new MILP must be novel and feature different constraints, variables, objectives, data, and parameters than the given MILP. 
The resulting new MILP should be a more complex version from the given MILP to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
Be creative, but ensure the new MILP is realistic and models genuine optimization challenges.
"""


PROMPT1_N = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP following the given MILP code structure:

1. Provide a Detailed Description of the python coding style of the given MILP code.

2. Provide a Detailed Description of new MILP code to suite a different optimization problem with a different real world application.
    - Discuss the novelty of the new MILP code in terms of constraints, variables, objective function, data and parameters and how they align with a specific real-world scenario.
    - The names of the new constraints, variables, or data should be full words starting with one of the letter {random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)}.

3. Generate the new MILP code by following the coding style of the given MILP code. For example, it must contain the following components """ + PROMPT_SYNTAX_NEW.format(prompt_optim="reflect the new real world application",
                                    prompt_data="support the new constraints, variables or objective", prompt_param="support the data generation for the optimization") + PROMPT_REQ + f"""
- Avoid Repeating Existing Code. Make the modified constraints, variables, objectives, data, and parameters distinct from the original.

Output Format:

### Description of the python coding style of the given MILP:
    <text to explain of the python coding style>

### Description of the new MILP:
    Names: <The full words of the names of constraint/variable/objective/data/parameter which starts with the capital letter selected from above.>
    <text to explain the new MILP and its real-world application>

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```
    
Here is the given MILP Code:
""" + "{code}"

