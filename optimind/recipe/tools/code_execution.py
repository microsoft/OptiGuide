import re
import subprocess
import sys
import tempfile
import uuid

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4


from recipe.tools.base_tool import BaseTool
from recipe.tools.schemas import OpenAIFunctionToolSchema
from recipe.tools.rollout_traces import rollout_trace_op 

# sys import parent's parent directory utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from rewards.utils import extract_python_code_block, run_full_code, compute_acc

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
########################################################################################

class CodeExecutionTool(BaseTool):
    """A demo tool for executing the code and return the standard output and error

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_execution",
                "description": "A tool for executing the {solver} code and return the standard output and error",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The {solver} code to be executed",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self.max_assistant_turns = config.get("max_assistant_turns", 3)  
        self.tool_format = config.get("tool_format", "interaction")
        
        if self.tool_format == "tool":
            self.tool_str = """Based on the output and error message (if any), please think step by step to determine if the code is correct. 
Based on your thoughts, please invoke a `code_execute` function call with an (improved) {solver} code if you have identified mistakes in the previous code.
If the previous code is correct, you do not need to invoke the `code_execute` tool again.

For each function call, Do not directly return the code in the ```python ``` block, as it will not be detected by the system. 
Instead, you should return a json object with function name and arguments by making the appropriate tool call.
"""
        elif self.tool_format == "minimal":
            self.tool_str = ""
        else:
            self.tool_str = """Based on the output and error message (if any), please think step by step to determine if the code is correct. 
Based on your thoughts, please provide an (improved) {solver} code in the ```python\n``` code block if you have identified mistakes in the previous code.
If the previous code is correct, you do not need to provide a new code block again. 
Your performance will be evaluated based on the correctness of the final ```python\n``` code block.
"""
        
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "round": 0, 
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        code_response = parameters.get("code", "")
        if not isinstance(code_response, str):
            code_response = str(code_response)

        self._instance_dict[instance_id]["response"] = code_response

        ## update one step
        self._instance_dict[instance_id]["round"] += 1
        cur_round = self._instance_dict[instance_id]["round"]
        solver = kwargs.get("solver", "gurobipy")
        results = await self.calc_reward(instance_id, precision=kwargs.get("precision", None),
                                         solver=solver)
        reward, output, error = results["score"], results["output"], results["traceback_str"]

        # penalty for non improved answer submission
        # tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        tool_reward = self._instance_dict[instance_id]["reward"] - reward
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward

        metrics = {"score": reward, "no_execution_error": results["no_execution_error"],
                   "have_objective": results["have_objective"], 
                   "objective": 0.0 if results["objective"] is None else float(results["objective"]),
                   "accuracy": results["accuracy"] if "accuracy" in results else 0.0}
        
        return_str = f"""The standard output and error message (if any) for your generated code are as follows:
```
[STDOUT] {output}
[STDERR] {error}
```
"""
        if cur_round  < self.max_assistant_turns:
            # add the suffix to ask the model to generate the next code, if not the last round
            return_str += self.tool_str.format(solver=solver)
        
        return return_str, tool_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # find the code block in the content 
        response = self._instance_dict[instance_id]["response"]
        ground_truth = self._instance_dict[instance_id]["ground_truth"]

        code_block = extract_python_code_block(response)
        solver = kwargs.get("solver", "gurobipy")
        solution_dict = run_full_code(code_block, solver=solver)
        objective = solution_dict.get("objective_value", None)

        if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
            # if any is correct then correct
            score = float(any([compute_acc(objective, gt, precision=kwargs.get("precision", None)) for gt in ground_truth]))
        else:
            score = float(compute_acc(objective, ground_truth, precision=kwargs.get("precision", None)))

        no_execution_error = float("failed" not in solution_dict["output"].lower() and solution_dict["traceback_str"].strip() == "" )
        have_objective = float(objective is not None)

        return {
            "score": score,
            "accuracy": score,
            "no_execution_error": no_execution_error,
            "have_objective": have_objective,
            "objective": objective,
            "output": solution_dict["output"],
            "traceback_str": solution_dict["traceback_str"]
        }

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
        