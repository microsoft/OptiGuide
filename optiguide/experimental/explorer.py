import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from hashlib import md5
from typing import Dict, List, Literal, Optional, Tuple, Union

from autogen import Agent, AssistantAgent, UserProxyAgent
from autogen.code_utils import _cmd, content_str
from autogen.oai import OpenAIWrapper
from termcolor import colored

try:
    import docker
except ImportError:
    docker = None

##########
CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "extensions")
UNKNOWN = "unknown"
TIMEOUT_MSG = "Timeout"
DEFAULT_TIMEOUT = 600
WIN32 = sys.platform == "win32"
PATH_SEPARATOR = WIN32 and "\\" or "/"

logger = logging.getLogger(__name__)

########## Constants Setup ##########
# Ignored with the AutoGen implementation, where privacy is not considered yet.

# SAFE_FLAG = "<SAFE>"
# SUPPORTED_CMDS = ["cd", "ls", "cat", "head", "echo", "python", "pip", "exit"]
# # Common programming language suffixes
# CODE_SUFFIXES = (".py", ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx",
#                  ".cs", ".java", ".go")

# # Common data file suffixes
# DATA_SUFFIXES = (".csv", ".tsv", ".json")

# # Common text file suffixes
# TEXT_SUFFIXES = (".txt", ".md")

########## System Message ##########
DEFAULT_SYSTEM_MESSAGE = """
You are a helpful AI assistant to help code editing in a large code repo.
You should answer my questions by using data and documents inside the repository.
You can explore the code repo by sending me system commands: ls, cd, cat, and python.

The tools you can use
1. Read files by using `cat`.
2. You can perform one action to avoid memory and space limits.
3. List all files with `ls`.
4. Change directory to a folder with `cd`.
5. Write Python code in code block, which will be executed and print messages will be
    returned.

--- Use the format ---
REASON: explain why you want to perform an action
ACTION:
```sh
YOU CODE GOES HERE
```

You should provide only one set of code in each reply, because I can not run many
things at once.
------

Note that:
1. Initially, you are at the root of the repo.
2. You can ONLY use linux commands: cd, ls, cat, and python.
3. I can run your commands and Python code, but I don't have intelligence to answer
    your questions.
4. You are all by yourself, and you need to explore the code repo by yourself.
5. You can use various techniques here, such as summarizing a book, thinking about
    code logic, architecture design, and performing analyses.
6. Feel free to use any other abilities, such as planning, executive functioning, etc.
7. You may need to cross reference different files!
8. If you can not access any folder, it is probably you are at the wrong location. Use
    `pwd` to check your current location.

When everything is done, reply "TERMINATE" as the single word in response.


----- tree structure of directories in the repo ------
{all_files}
"""

CONSTRUCTION_QUESTIONS = [
    "Read important files, understand the repo, and summarize the "
    "knowledge in 500 words.",
    "What programming languages are mainly used in this project?",
]


########## Exploration Agent ##########
class ProjectAgent(AssistantAgent):
    """
    The project agent is a special agent that helps the user explore the code repo.
    """

    def __init__(self,
                 name,
                 root,
                 max_consecutive_auto_reply=10,
                 llm_config: Optional[Union[Dict, Literal[False]]] = None,
                 code_execution_config: Optional[Dict] = None,
                 description: Optional[str] = "",
                 **kwargs):
        self.root = root

        _system_message = DEFAULT_SYSTEM_MESSAGE.format(
            root=os.path.basename(root),
            root2=os.path.basename(root),
            all_files=display_files_recursively(root))

        # is_termination_msg = lambda x: content_str(x).strip().rstrip() == "exit"

        super().__init__(
            name=name,
            system_message=_system_message,
            # is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode="NEVER",
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        self.env = RepositoryEnv(
            repository_root=root,
            # TODO: add back the support of private files in the
            # future.
            code_execution_config=code_execution_config)
        self.register_reply([Agent, None], ProjectAgent._repo_generate_reply)

        self._warm_up()

    def _warm_up(self):
        for msg in CONSTRUCTION_QUESTIONS:
            self.env.initiate_chat(self, message=msg, clear_history=False)

    def _repo_generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""

        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        if not isinstance(sender, RepositoryEnv):
            # Not the sub-module. So, we delegate to the environment.
            self.env.initiate_chat(self,
                                   message=messages[-1],
                                   clear_history=False)
            _messages = self._oai_messages[self.env]

            # Find the answer reversely
            for i in range(len(_messages) - 1, -1, -1):
                if content_str(_messages[i]["content"]) != "TERMINATE":
                    return True, content_str(_messages[i]["content"])

            # Unable to retrieve the answer, because all messages are TERMINATE.
            return False, None
        else:
            # the sender is the environment, and we need standard generate_reply (e.g.,
            # generate_oai_reply).
            return False, None


########## Environments ##########
class RepositoryEnv(UserProxyAgent):
    """
    The RepositoryEnv is an UserProxyAgent with special safety checks and
    commands handling.

    1. It will clone a repository, perform actions inside the clone.
    2. If will clean up the clone when the environment is removed. Modified
        files will be  saved in a dictionary for future reference, which is
        great for creating pull requests or perform code review.
    3. It will keep track of the current working directory, and also hide the
       prefix of the temporary clone.
    """

    def __init__(
        self,
        repository_root: str,
        code_execution_config: Optional[Dict] = {},
    ):
        """
        Wraps the dataset folder.
        Prevents any system commands from ruining the original dataset.

        Args:
        - repository_root (str): The path to the dataset.
        """

        # Override the "work_dir" with sandbox folder.
        if code_execution_config is None:
            code_execution_config = {}
        _working_dir = code_execution_config.get("work_dir", ".")
        _working_dir = _working_dir.replace("\\", "/")
        os.makedirs(_working_dir, exist_ok=True)
        self.repo_path = os.path.abspath(repository_root).replace("\\",
                                                                  "/") + "/"

        # Copy dataset to a temporary directory in the working directory
        # Also, normalize the windows file path
        self.sandbox_dir = tempfile.mkdtemp(dir=_working_dir).replace(
            "\\", "/")
        self.sandbox_dir = os.path.abspath(self.sandbox_dir).rstrip("/")
        code_execution_config["work_dir"] = self.sandbox_dir
        code_execution_config["use_docker"] = False

        try:
            shutil.copytree(self.repo_path,
                            self.sandbox_dir,
                            ignore=shutil.ignore_patterns('.*', '*.pyc'),
                            dirs_exist_ok=True)
        except Exception as e:
            raise Exception(
                f"Failed to copy data from {self.repo_path} to {self.sandbox_dir}"
            ) from e

        # Checkpoint cwd to avoid outside changes
        # self.cwd = self.sandbox_dir
        print(
            colored(f"Data copied to temporary directory: {self.sandbox_dir}",
                    "green"))

        super().__init__(name="Repository Environment Agent",
                         system_message="",
                         max_consecutive_auto_reply=1000000,
                         human_input_mode="NEVER",
                         function_map=None,
                         code_execution_config=code_execution_config,
                         llm_config=False,
                         default_auto_reply="TERMINATE",
                         description="")

    def __del__(self):
        # Upon deletion, clean up the temporary directory
        # If it is windows, run 'rmdir'
        # otherwise, run rm -rf
        if os.name == "nt":
            # Windows
            os.system('rmdir /S /Q "{}"'.format(self.sandbox_dir))
        else:
            # Unix
            os.system('rm -rf "{}"'.format(self.sandbox_dir))

    def _get_loc(self, **kwargs):
        code = "\npwd\n"
        kwargs.pop("lang", None)
        exit_code, logs, image = execute_code(code, lang="sh", **kwargs)
        loc = os.path.relpath(logs.strip().rstrip(), self.sandbox_dir)
        import pdb
        pdb.set_trace()
        return f"\n\n\nSide Note: current location is at `{loc}`"

    # Override AutoGen ConversableAgent's run_code
    def run_code(self, code, **kwargs):
        """Run the code and return the result.

        Override this function to modify the way to run the code.
        Args:
            code (str): the code to be executed.
            **kwargs: other keyword arguments.

        Returns:
            A tuple of (exitcode, logs, image).
            exitcode (int): the exit code of the code execution.
            logs (str): the logs of the code execution.
            image (str or None): the docker image used for the code execution.
        """
        lang = kwargs.get("lang", None)
        if lang in ["bash", "shell", "sh"]:
            kwargs["lang"] = "sh"
            # code += """\n\necho Side Note: Current Dir is "$(pwd)"\n"""
            exit_code, logs, image = execute_code(code, **kwargs)
            # logs += self._get_loc(**kwargs)
        else:
            exit_code, logs, image = execute_code(code, **kwargs)

        logs = logs.replace(self.sandbox_dir, "")
        return exit_code, logs, image


def display_files_recursively(
    folder_path: str,
    indent: str = "",
    file_suffixes: list = [".py", ".cpp", ".cs", ".md", ".txt", ".csv"],
) -> str:
    """Recursively lists files with specific suffixes from a given directory
      and its subdirectories.

    This function searches through the directory structure starting from
    `folder_path` and returns a string representation of the directory
    hierarchy containing only the files that match any of the provided
     `file_suffixes`. Each level in the hierarchy increases
    the indentation in the returned string, enhancing readability.

    Args:
    - folder_path (str): Path to the starting folder from which the
        search begins.
    - indent (str, optional): The indentation string for the current level
        of the directory hierarchy. Defaults to an empty string.
        Typically used internally for recursive calls.
    - file_suffixes (list of str, optional): A list of file suffixes
        (extensions) to consider while listing files. Defaults to [".py",
        ".cpp", ".cs", ".md", ".txt"].

    Returns:
    - str: A string representation of the directory hierarchy with files
        matching the provided suffixes. Directories are only displayed if they
        contain at least one valid file or subdirectory with valid files.

    Note:
    - The function assumes that the provided `folder_path` exists and is
        accessible.
    """
    ret = ""

    valid_files = [
        file_name for file_name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file_name))
        and file_name.endswith(tuple(file_suffixes))
    ]

    # Display valid files in the current folder
    for file_name in valid_files:
        if ret == "":
            ret += "\n" + indent + os.path.basename(folder_path) + "/"
        ret += "\n" + indent + "    " + file_name

    # Recurse into directories
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            # Recursively check if sub-directory contains valid files or folders
            # with valid files
            ret += display_files_recursively(dir_path, indent + "    ",
                                             file_suffixes)

    return ret


def execute_code(
    code: Optional[str] = None,
    timeout: Optional[int] = None,
    filename: Optional[str] = None,
    work_dir: Optional[str] = None,
    use_docker: Optional[Union[List[str], str, bool]] = None,
    lang: Optional[str] = "python",
) -> Tuple[int, str, str]:
    if all((code is None, filename is None)):
        error_msg = f"Either {code=} or {filename=} must be provided."
        logger.error(error_msg)
        raise AssertionError(error_msg)

    if use_docker and docker is None:
        error_msg = ("Cannot use docker because the python docker package "
                     "is not available.")
        logger.error(error_msg)
        raise AssertionError(error_msg)

    # Warn if use_docker was unspecified (or None), and cannot be provided
    # (the default).
    # In this case the current behavior is to fall back to run natively,
    # but this behavior
    # is subject to change.
    if use_docker is None:
        if docker is None:
            use_docker = False
            logger.warning(
                "execute_code was called without specifying a value for use_docker. "
                "Since the python docker package is not available, code will be "
                "run natively. Note: this fallback behavior is subject to change"
            )
        else:
            # Default to true
            use_docker = True

    timeout = timeout or DEFAULT_TIMEOUT
    original_filename = filename
    if WIN32 and lang in ["sh", "shell"] and (not use_docker):
        lang = "ps1"
    if filename is None:
        code_hash = md5(code.encode()).hexdigest()
        # create a file with a automatically generated name
        filename = (f".tmp_code_{code_hash}"
                    f".{'py' if lang.startswith('python') else lang}")
    if work_dir is None:
        work_dir = WORKING_DIR
    filepath = os.path.join(work_dir, filename)
    file_dir = os.path.dirname(filepath)
    os.makedirs(file_dir, exist_ok=True)
    if code is not None:
        with open(filepath, "w", encoding="utf-8") as fout:
            fout.write(code)
    # check if already running in a docker container
    in_docker_container = os.path.exists("/.dockerenv")
    if not use_docker or in_docker_container:
        # already running in a docker container
        cmd = [
            sys.executable if lang.startswith("python") else _cmd(lang),
            f".\\{filename}" if WIN32 else filename,
        ]
        if WIN32:
            logger.warning(
                "SIGALRM is not supported on Windows. No timeout will be enforced."
            )
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    subprocess.run,
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                try:
                    result = future.result(timeout=timeout)
                except TimeoutError:
                    if original_filename is None:
                        os.remove(filepath)
                    return 1, TIMEOUT_MSG, None
        if original_filename is None:
            os.remove(filepath)
        if result.returncode:
            logs = result.stderr
            if original_filename is None:
                abs_path = str(pathlib.Path(filepath).absolute())
                logs = logs.replace(str(abs_path), "").replace(filename, "")
            else:
                abs_path = str(
                    pathlib.Path(work_dir).absolute()) + PATH_SEPARATOR
                logs = logs.replace(str(abs_path), "")
        else:
            logs = result.stdout
        return result.returncode, logs, None

    # create a docker client
    client = docker.from_env()
    image_list = ([
        "python:3-alpine", "python:3", "python:3-windowsservercore"
    ] if use_docker is True else
                  [use_docker] if isinstance(use_docker, str) else use_docker)
    for image in image_list:
        # check if the image exists
        try:
            client.images.get(image)
            break
        except docker.errors.ImageNotFound:
            # pull the image
            print("Pulling image", image)
            try:
                client.images.pull(image)
                break
            except docker.errors.DockerException:
                print("Failed to pull image", image)
    # get a randomized str based on current time to wrap the exit code
    exit_code_str = f"exitcode{time.time()}"
    abs_path = pathlib.Path(work_dir).absolute()
    cmd = [
        "sh",
        "-c",
        f"{_cmd(lang)} {filename}; exit_code=$?; echo -n {exit_code_str}; "
        f"echo -n $exit_code; echo {exit_code_str}",
    ]
    # create a docker container
    container = client.containers.run(
        image,
        command=cmd,
        working_dir="/workspace",
        detach=True,
        # get absolute path to the working directory
        volumes={abs_path: {
            "bind": "/workspace",
            "mode": "rw"
        }},
    )
    start_time = time.time()
    while container.status != "exited" and time.time() - start_time < timeout:
        # Reload the container object
        container.reload()
    if container.status != "exited":
        container.stop()
        container.remove()
        if original_filename is None:
            os.remove(filepath)
        return 1, TIMEOUT_MSG, image
    # get the container logs
    logs = container.logs().decode("utf-8").rstrip()
    # commit the image
    tag = filename.replace("/", "")
    container.commit(repository="python", tag=tag)
    # remove the container
    container.remove()
    # check if the code executed successfully
    exit_code = container.attrs["State"]["ExitCode"]
    if exit_code == 0:
        # extract the exit code from the logs
        pattern = re.compile(f"{exit_code_str}(\\d+){exit_code_str}")
        match = pattern.search(logs)
        exit_code = 1 if match is None else int(match.group(1))
        # remove the exit code from the logs
        logs = logs if match is None else pattern.sub("", logs)

    if original_filename is None:
        os.remove(filepath)
    if exit_code:
        logs = logs.replace(
            f"/workspace/{filename if original_filename is None else ''}", "")
    # return the exit code, logs and image
    return exit_code, logs, f"python:{tag}"
