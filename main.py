import os
import argparse
import requests
from enum import Enum
from dotenv import load_dotenv
from contextlib import redirect_stdout
from io import StringIO
import re

import openai
import autogen
import gradio as gr
from autogen.agentchat import UserProxyAgent
from optiguide.optiguide import OptiGuideAgent

# Constants
SUCCESS_CODE = 200
API_KEY_ENV_VAR = 'api_key'
DEFAULT_CODE_FILE = 'data/benchmark/workforce5.py'
DEFAULT_MODEL_TYPE = 'gpt-4-1106-preview'

# Enums for clarity
class HumanInputMode(Enum):
    NEVER = "NEVER"
    # Add other modes if necessary

def load_env_variables():
    load_dotenv()
    api_key = os.getenv(API_KEY_ENV_VAR)
    openai.api_key = api_key

def config_parser():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--code_file', type=str, default=DEFAULT_CODE_FILE, help='')
    parser.add_argument('--debug_times', type=int, default=3, help='')
    parser.add_argument('--max_consecutive_auto_reply', type=int, default=0, help='')
    parser.add_argument('--human_input_mode', type=HumanInputMode, default=HumanInputMode.NEVER, choices=list(HumanInputMode), help='')
    parser.add_argument('--code_execution_config', type=bool, default=False, help='')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, help='')
    return parser.parse_args()

def load_code_from_url(code_url: str) -> (str, str):
    response = requests.get(code_url)
    if response.status_code != SUCCESS_CODE:
        raise RuntimeError("Failed to retrieve the file.")
    name = code_url.split('.')[-2]
    return name, response.text

def load_code_from_file(code_file: str) -> (str, str):
    name = code_file.split('.')[-2]
    with open(code_file, 'r') as file_code:
        return name, file_code.read()

def load_code(args) -> (str, str):
    if args.code_file and args.code_url:
        raise ValueError("Specify either code_file or code_url, not both.")
    if args.code_url:
        return load_code_from_url(args.code_url)
    return load_code_from_file(args.code_file)

def initialize_agent(name: str, code: str, config_list: list, debug_times: int) -> OptiGuideAgent:
    return OptiGuideAgent(
        name=name,
        source_code=code,
        debug_times=debug_times,
        example_qa="",
        llm_config={
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
        }
    )

def chat_with_agent(agent: OptiGuideAgent, user: UserProxyAgent, message: str, history: str) -> str:
    agent._example_qa = ""
    output_capture = StringIO()
    with redirect_stdout(output_capture):
        user.initiate_chat(agent, message=message, clear_history=False, history=history)
    output = output_capture.getvalue()
    output_user = [match.end() for match in re.finditer(r'\(to user\):', output)]
    return output[output_user[-1]:]

def main():
    load_env_variables()
    args = config_parser()
    autogen.oai.ChatCompletion.start_logging()
    config_list = [{'model': args.model_type}]

    name, code = load_code(args)
    agent = initialize_agent(name, code, config_list, args.debug_times)

    user = UserProxyAgent(
        "user",
        max_consecutive_auto_reply=args.max_consecutive_auto_reply,
        human_input_mode=args.human_input_mode,
        code_execution_config=args.code_execution_config
    )

    demo = gr.ChatInterface(lambda message, history: chat_with_agent(agent, user, message, history))
    demo.launch()

if __name__ == '__main__':
    main()
