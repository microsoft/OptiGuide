import os
import random
import shutil
import string
import subprocess
import tempfile
from hashlib import sha256
from typing import Dict, Literal, Optional, Union

import tiktoken
from autogen import AssistantAgent, UserProxyAgent
from termcolor import colored

########## Constants Setup ##########
SAFE_FLAG = "<SAFE>"
SUPPORTED_CMDS = ["cd", "ls", "cat", "head", "echo", "python", "pip", "exit"]
# Common programming language suffixes
CODE_SUFFIXES = (".py", ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx",
                 ".cs", ".java", ".go")

# Common data file suffixes
DATA_SUFFIXES = (".csv", ".tsv", ".json")

# Common text file suffixes
TEXT_SUFFIXES = (".txt", ".md")

########## System Message ##########
DEFAULT_SYSTEM_MESSAGE = """
You are a helpful AI assistant to help code editing in a large code repo.
You can explore the code repo by sending me system commands: ls, cd, cat, and echo.

The tools you can use
1.  Read files by using `cat`.
2. You can only read one file a time to avoid memory and space limits,
    and you should avoid reading a file multiple times.
3.  Write memory files by using `echo`.
4.  List all files with `ls`.
5.  Change directory to a folder with `cd`.

--- Use the format ---
REASON: explain why you want to perform an action
ACTION:
```bash
YOU CODE GOES HERE
```
------

Note that:
1.  Initially, you are at the root of the repo.
2. You can ONLY use linux commands: cd, ls, cat, echo
3. I can run your commands, but I don't have intelligence to answer your questions.
4. You are all by yourself, and you need to explore the code repo by yourself.
5. You can use various techniques here, such as summarizing a book, thinking about
    code logic, architecture design, and performing analyses.
6. Feel free to use any other abilities, such as planning, executive functioning, etc.
7. You may need to cross reference different files!

----- tree structure of directories in the repo ------
{all_files}
"""

CONSTRUCTION_QUESTIONS = [
    ("Read the repo, understand what it means and all its files. "
     "Then, summarize the knowledge in SUMMARY.txt"),
]


########## Exploration Agent ##########
class ProjectAgent(AssistantAgent):
    """
    The exploration agent is a special agent that helps the user explore the code repo.

    """

    def __init__(self,
                 name,
                 root,
                 data_path,
                 max_consecutive_auto_reply=10,
                 llm_config: Optional[Union[Dict, Literal[False]]] = None,
                 description: Optional[str] = "",
                 **kwargs):

        system_message = open("prompt_templates/explore_prompt.md", "r").read()
        system_message = system_message.format(
            root=os.path.basename(root),
            root2=os.path.basename(root),
            all_files=display_files_recursively(root))

        is_termination_msg = ""

        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode="NEVER",
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        self.data_path = data_path

        for msg in self.msgs:
            print(colored_string(msg))

        os.chdir(root)


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
        dataset_path: str,
        password: str,
        private_files: list = [],
    ):
        """
        Wraps the dataset folder.
        Prevents any system commands from ruining the original dataset.

        Args:
        - dataset_path (str): The path to the dataset.
        """
        # Use absolute path
        self.working_dir = os.path.abspath(".").replace("\\", "/") + "/"
        self.dataset_path = os.path.abspath(dataset_path).replace("\\",
                                                                  "/") + "/"

        # Copy dataset to a temporary directory in the working directory
        # Also, normalize the windows file path
        self.sandbox_dir = tempfile.mkdtemp(dir=self.working_dir).replace(
            "\\", "/") + "/"

        # Store the hashed password for identity verification
        self._sandbox_id = ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=10))
        self._hashed_password = self._hash_password(password)
        self.private_files = private_files

        # Ignore hidden files and directories
        def ignore(directory, filenames):
            return [fn for fn in filenames if fn.startswith('.')]

        shutil.copytree(self.dataset_path,
                        self.sandbox_dir,
                        ignore=ignore,
                        dirs_exist_ok=True)
        print(
            colored(f"Data copied to temporary directory: {self.sandbox_dir}",
                    "green"))

        # Checkpoint cwd to avoid outside changes
        self.cwd = self.sandbox_dir

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

    def _hash_password(self, password: str) -> str:
        return sha256(
            (password + self._sandbox_id).encode("utf-8")).hexdigest()

    def safety_check(self, cmd: list, password: str) -> str:
        """
        Return "SAFE" iff the cmd is safe to run.
        Otherwise, return error message.

        Args:
        - cmd (list): a single command splitted into a list of arguments.
        - password (str): the password for identity verification.

        Returns:
        """
        # First check if password is correct
        if self._hash_password(password) != self._hashed_password:
            return "Error: Wrong password!"

        # Restrict command type
        if cmd[0] == "exit":
            raise NotImplementedError(
                "exit should be handled outside of run_command().")
        if cmd[0] not in SUPPORTED_CMDS:
            return f"Error: You can only use {', '.join(SUPPORTED_CMDS[:-1])}."

        # Test if the target file/dir is inside the sandbox
        target_dirs = get_target_dirs(cmd)
        for target_dir in target_dirs:
            if "Error" in target_dir:
                return target_dir
            if not target_dir.startswith(self.sandbox_dir):
                return (
                    f"Error: You cannot access file {target_dir} "
                    f"outside the repo! You are now at {self._get_relative_cwd()}"
                )

        # Check if the target file is private
        files = get_file_names(cmd)
        for file in files:
            if file in self.private_files:
                return f"Error: You cannot access a private file {file}!"

        return SAFE_FLAG

    def run_command(self, cmd: list, password: str) -> str:
        """Wrapper function for self._run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        "cd", "ls", "cat", "head", "tail", "echo", "python", "pip"
        "exit" is handled outside of this function.

        Args:
        - cmd (list): a single command splitted into a list of arguments.
        - password (str): the password for identity verification.

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """
        # Restore to the checkpointed cwd
        _cwd = os.getcwd()
        os.chdir(self.cwd)

        safety_check_result = self.safety_check(cmd, password)

        if safety_check_result != SAFE_FLAG:
            ret = safety_check_result
        else:
            ret = self._run_command(cmd)

        # Checkpoint cwd
        self.cwd = os.getcwd().replace("\\", "/") + "/"
        os.chdir(_cwd)

        return ret

    def _run_command(self, cmd: list) -> str:
        """Inner function for self.run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        "cd", "ls", "cat", "head", "tail", "echo", "python", "pip".
        "exit" is handled outside of this function.

        Args:
        - cmd (list): a single command splitted into a list of arguments.

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """

        # Check if echo outputs to a file
        if cmd[0] == "echo" and len(cmd) == 3:
            return "Warning: echo command without output file, ignored."

        # Run the command
        try:
            if cmd[0] == "cd":
                # cd cannot be handled by subprocess
                os.chdir(cmd[1])
                return "Success: Now at " + self._get_relative_cwd()
            else:
                result = subprocess.run(' '.join(cmd),
                                        shell=True,
                                        capture_output=True)
                return self.respond_cmd(cmd, result)
        except Exception as e:
            return "Error: " + str(e)

    def respond_cmd(self, cmd: list, result) -> str:
        """
        Generate the response for the result of a command.

        Args:
        - cmd (list): a single command splitted into a list of arguments.
        - result (subprocess.CompletedProcess): the result of the command.

        Returns:
        - str: the response for the result of the command.
        """
        rstdout = result.stdout.decode('utf-8')
        rstderr = hide_root(result.stderr.decode('utf-8'), self.sandbox_dir)

        if cmd[0] == "ls":
            return "Success: The result of ls is:\n" + rstdout
        elif cmd[0] in ["cat", "head", "tail"]:
            fn = get_file_names(cmd)[0]
            return (f"Success: The content of {fn} is:\n" +
                    trunc_text(fn, rstdout))
        elif cmd[0] == "echo":
            return f"Success: echoed to {cmd[-1]}"
        elif cmd[0] == "python":
            if rstderr != "":
                return f"Error: {rstderr}"
            else:
                return f"Success: The output of python is:\n{rstdout}"
        elif cmd[0] == "pip":
            if rstderr != "":
                return f"Error: {rstderr}"
            else:
                return "Success: pip succeeded"
        else:
            raise NotImplementedError(f"Does not support command: {cmd[0]}")

    def get_changed_files(self) -> dict:
        """
        Return the name and content of changed files in the sandbox.

        Returns:
        - dict: key is relative file path, value is the content in bytes.
        """
        original_files = set(list_files(self.dataset_path))
        current_files = set(list_files(self.sandbox_dir))
        changed_files = list(current_files - original_files)

        common_files = current_files.intersection(original_files)

        for file in common_files:
            file = file.replace("\\", "/")

            original_file_path = self.dataset_path + file
            current_file_path = self.sandbox_dir + file

            original_file_content = open(original_file_path, "rb").read()
            current_file_content = open(current_file_path, "rb").read()

            if original_file_content != current_file_content:
                changed_files.append(file)

        print(colored("List of changed files:", "yellow"))
        print(changed_files)

        return {
            file: open(self.sandbox_dir + file, "rb").read()
            for file in changed_files
        }

    def _get_relative_cwd(self):
        "Return the relative path to the sandbox's root directory."
        return os.path.relpath(os.getcwd().replace('\\', '/'),
                               self.sandbox_dir) + "/"


########## Utils
def trunc_text(file: str, content: str) -> str:
    """
    Truncate the content of a file for `cat`, `head`, `tail` command if it is too long.
    It will truncate to a maximum line and a maximum token, depending on the file type.

    Args:
    - file_name (str): The name of the file.
    - content (str): The content of the file.

    Returns:
    - str: The truncated content.
    """

    # Define truncate function
    def _trunc_text(content: str, max_line: int, max_token: int) -> str:
        """
        Truncate the content of a file for `cat`, `head`, `tail` command
        if it is too long.
        Truncate to `max_line` lines or `max_token` tokens, whichever is smaller.

        Args:
        - file_name (str): The name of the file.
        - content (str): The content of the file.
        - max_line (int): The maximum number of lines to display.
        - max_token (int): The maximum number of tokens to display.

        Returns:
        - str: The truncated content.
        """
        truncated = False

        lines = content.split("\n")
        if len(lines) > max_line:
            content = "\n".join(lines[:max_line])
            truncated = True

        encoder = tiktoken.encoding_for_model("gpt-4")
        encoded = encoder.encode(content)
        if len(encoded) > max_token:
            content = encoder.decode(encoded[:max_token])
            truncated = True

        if truncated:
            content += ("\n...\nLarge file, only display first "
                        f"{max_line} lines and {max_token} tokens.\n")

        return content

    if file[-1] in ['"', "'"]:
        file = file[1:-1]

    # Truncate the content depending on file type
    if file.endswith(CODE_SUFFIXES):
        return _trunc_text(content, 1000, 1000)
    elif file.endswith(DATA_SUFFIXES):
        return _trunc_text(content, 5, 500)
    elif file.endswith(TEXT_SUFFIXES):
        return _trunc_text(content, 100, 1000)
    else:
        return _trunc_text(content, 10, 1000)


def colored_string(msg: str) -> str:
    color_dict = {"system": "blue", "user": "green", "assistant": "cyan"}
    return colored(msg[1], color_dict[msg[0]])


def list_files(directory: str, ignore_hidden: bool = True) -> list:
    """
    List all files in a directory (recursively).

    Args:
    - directory (str): The path to the directory to list files from.
    - ignore_hidden (bool, optional): Whether to ignore hidden files.
        Defaults to True.

    Returns:
    - list of str: A list of file paths relative to the input directory.
    """
    for root, dirs, files in os.walk(directory):
        if ignore_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if ignore_hidden and file.startswith("."):
                continue
            yield os.path.relpath(os.path.join(root, file), directory)


def hide_root(text, root) -> str:
    """
    Hide all root paths in a text.

    Args:
    - text (str): The text to replace paths in.
    - root (str): The root path.

    Returns:
    - str: The text with all root paths hidden.
    """
    # Regular expression pattern to match absolute file paths.
    # This pattern assumes that paths start with / followed by any non-space characters.
    text = text.replace(root, "")
    text = text.replace(root[:-1], ".")
    return text


def get_target_dirs(cmd: list) -> list:
    """
    Get the directory of the target file/dir from a command.

    Args:
    - cmd (list): a single command splitted into a list of arguments.

    Returns:
    - list: A list of the directories of the target file/dirs.
            If error occurs, return a list of error messages.
    """
    # Get the files
    files = get_file_names(cmd)
    target_dirs = []

    for file in files:
        path = os.path.dirname(file) if "." in os.path.basename(file) else file
        if path == "":
            path = "."

        # Backup the cwd
        original_cwd = os.getcwd()

        try:
            os.chdir(path)
        except Exception as e:
            return ["Error: " + str(e)]

        target_dirs.append(os.getcwd().replace('\\', '/') + "/")

        # Restore the cwd
        os.chdir(original_cwd)

    return target_dirs


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


def find_all_substr(string, substr):
    """
    Finds all occurrences of a substring in a string and returns their starting
    indices.

    This function scans the input string for all instances of the specified
    substring and returns a list of indices where these instances start. If the
    substring is not found in the string, an empty list is returned.

    Args:
    - string (str): The input string in which to search for the substring.
    - substr (str): The substring to search for.

    Returns:
    - list of int: A list of starting indices where the substring is found. The
         list is empty if the substring is not found.
    """
    start_index = 0
    positions = []

    while True:
        index = string.find(substr, start_index)
        if index == -1:
            break
        positions.append(index)
        start_index = index + 1

    return positions


def get_file_names(command: list) -> list:
    """
    Extract file names from the command.

    Args:
    - command (list): The command splitted into a list.

    Returns:
    - list: A list of file names.
    """
    if command[0] == "ls":
        if len(command) > 1:
            return [command[1]]
        else:
            return ["."]
    elif command[0] == "cat":
        ret = [command[1]]
        if ">" in command or ">>" in command:
            ret.append(command[-1])
        return ret
    elif command[0] == "head":
        return [command[3]]
    elif command[0] == "cd":
        return [command[1]]
    elif command[0] == "echo":
        return [command[-1]]
    elif command[0] == "python":
        if command[2] == "-c":
            return ["."]
        else:
            for x in command:
                if x.endswith(".py"):
                    return [x]
    elif command[0] == "pip":
        return ["."]
    else:
        raise NotImplementedError(f"Does not support command: {command[0]}")
        raise NotImplementedError(f"Does not support command: {command[0]}")
