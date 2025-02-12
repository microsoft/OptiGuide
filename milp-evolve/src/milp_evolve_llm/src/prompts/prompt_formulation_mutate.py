from src.prompts.prompt_common import PROMPT_REQ, PROMPT_SYNTAX_MUTATE

SYSTEM_MESSAGE_FM = """
You are an expert in Mixed Integer Linear Programming (MILP), and you possess a deep understanding of MILP formulations. 
Your task is to transform a given MILP formulation into a new MILP that models a different real-world application. 
This new MILP should maintain a similar level of complexity as the original but a part of its constraints, variables, objectives, data, and parameters must be novel and different from the given MILP.
Be creative, but ensure the new MILP is realistic and models genuine optimization challenges.
"""


PROMPT1_FM = f"""
Follow these step-by-step instructions to generate a diverse and realistic MILP by slightly modifying the given MILP code, replacing some constraints into new constraints using a specific formulation method and slighlty modifying the variables and other components accordingly:

1. Provide a Detailed Description of the given MILP code:
    - Embed the MILP within a specific real-world application. 
    - Clearly describe the context, constraints, objective, and variables in a way that highlights its practical importance.

2. **Choose a MILP Formulation Method** from the following options: """ + "{formulation_methods}" + f""" 
    - Explain how we can replace some of the existing constraints in the given MILP code by new constraints using the chosen MILP formulation method in the real-world context.
    - Discuss whether we should (and if so, how to) slightly modify the variables, parameters, or data generation method to support the new constraints and make the new MILP coherent and realistic.

3. Generate the new MILP code by following these procedures step-by-step: """ + PROMPT_SYNTAX_MUTATE.format(prompt_optim="reflect the given MILP formulation method",
                            prompt_data="support the new constraints, variables or objective", prompt_param="support the data generation for the optimization") + PROMPT_REQ + f"""

Output Format:

### Description of the given MILP:
    <text to explain of the given MILP and its real-world application>

### Description of the modification of the given MILP using the chosen MILP formulation method:
    <text to explain which set of constraints in the given MILP should be replaced, what are the new constraints with the given formulation method, and whether and how should we change the data generation or parameter function to support the constraints replacement.>

### New complete MILP code:
    ```python
    <complete code that incorporates modified constraints, data generation, and parameters>
    ```
    
Here is the given MILP Code:
""" + "{code}"
