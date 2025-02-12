

import random
from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_MUTATE

SYSTEM_MESSAGE_M = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your task is to transform a given MILP formulation into a new MILP that models a different real-world application. 
This new MILP should maintain a similar level of complexity as the original but a part of its constraints, variables, objectives, data, and parameters must be novel and different from the given MILP.
Be creative, but ensure the new MILP is realistic and models genuine optimization challenges.
"""


PROMPT1_M = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP by slightly modifying the given MILP code:

1. Provide a Detailed Description of the given MILP code:
    - Embed the MILP within a specific real-world application. 
    - Clearly describe the context, constraints, objective, and variables in a way that highlights its practical importance.

2. Provide a Detailed Description of new MILP code to suite a different real world application.
    - Discuss the novelty of the modifying constraints, variables, or changing the objective function from the given MILP by embedding it in a different real-world scenario.
    - Explain the relevance of these changes, detailing the similarity and differences between the original and new MILP.
    - The names of the new constraints, variables, or data should be full words starting with one of the letter {random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)}.

3. Generate the new MILP code by following these procedures step-by-step: """ + PROMPT_SYNTAX_MUTATE.format(prompt_optim="reflect the new real world application",
                                prompt_data="support the new constraints, variables or objective", prompt_param="support the data generation for the optimization") + PROMPT_REQ + f"""

Output Format:

### Description of the given MILP:
    <text to explain of the given MILP and its real-world application>

### Description of the new MILP:
    Names: <The full words of the names of constraint/variable/objective/data/parameter which starts with the capital letter selected from above.>
    <text to explain the new MILP and its real-world application>

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```
    
Here is the given MILP Code:
""" + "{code}"

