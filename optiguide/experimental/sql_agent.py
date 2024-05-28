import re
import sqlite3
from typing import Dict, List, Optional

import tiktoken
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.code_utils import content_str
from termcolor import colored

############# Configs ##############
DEFAULT_SQL_CODE_CONFIG = {
    "last_n_messages": 3,
    "work_dir": ".",
    "use_docker": False
}


def IS_TERMINATE_MSG_FOR_SQL_ASSISTANT(x):
    content = content_str(x.get("content"))
    return content.find("ANSWER") >= 0


def IS_TERMINATE_MSG_FOR_SQL_PROXY(x):
    content = content_str(x.get("content"))
    return content.find("ANSWER") >= 0 and re.findall(r"```.+```", content,
                                                      re.DOTALL) == []


############# Agents ##############
class SQLAgent(ConversableAgent):

    def __init__(self, sql_file, name, max_sys_msg_tokens, llm_config: dict,
                 **kwargs):
        super().__init__(name,
                         llm_config=llm_config,
                         max_consecutive_auto_reply=0,
                         **kwargs)
        self.sql_file = sql_file
        connection = sqlite3.connect(sql_file)
        self.cursor = connection.cursor()

        self.assistant = AssistantAgent(
            name="sql llm",
            system_message=
            """You can interact with the SQLlite3 database and answer questions.

There are two types of answers you can provide: code and answer.

## Code
In the code format, you must write code directly in code block to answer questions.
For instance,
```python
# Python code by using the `cursor` variable.
```
The cursor variable is provided directly to you.
Save the result to the "rst" variable so that you can see the output.
Note that you are interacting with a Python environment, and it can only reply you
if you write code.


## Answer
If you gather enough information (e.g., from coding or prior knowledge) to answer the
user question, then you should reply with your final answer in the foramt:
ANSWER: your answer goes here.
""" + table_prompt(self.cursor,
                   question="",
                   model=llm_config["config_list"][0]["model"],
                   max_tokens=max_sys_msg_tokens),
            llm_config=llm_config,
            is_termination_msg=IS_TERMINATE_MSG_FOR_SQL_ASSISTANT,
            max_consecutive_auto_reply=10)

        self.proxy = SQLProxy(
            sql_file=sql_file,
            name="sql proxy",
            human_input_mode="NEVER",
            is_termination_msg=IS_TERMINATE_MSG_FOR_SQL_PROXY)
        self.register_reply([Agent, None], SQLAgent.generate_sql_reply)
        self.synth_history()

    def synth_history(self):
        # we inject some synthetic Q&A with the desired format,
        # so that the LLM can learn the pattern in context.
        tables = list_tables(self.cursor)
        n_tables = len(tables)
        query = f"SELECT COUNT(*) FROM {tables[0]};"
        n_rows_table0 = self.cursor.execute(query).fetchall()[0][0]
        table0_info = show_table_schema(self.cursor, tables[0])
        self.table_memory = {
            table_name: show_table_schema(self.cursor, table_name)
            for table_name in tables
        }

        prior = [
            {
                'content': 'QUESTION: How many tables are in the database?',
                'role': 'user'
            },
            {
                'content':
                'We can count the number of tables in the database.\n'
                '```python\ncursor.execute("SELECT COUNT(name) FROM '
                'sqlite_master WHERE type=\'table\';")\n'
                'rst = cursor.fetchone()\nrst\n```',
                'role':
                'assistant'
            },
            {
                'content':
                "exitcode: 0 (execution succeeded)\nCode output: \n{'rst': (" +
                str(n_tables) + ",)}",
                'role':
                'user'
            },
            {
                'content': f'ANSWER: The database contains {n_tables} tables. '
                'If you need any more information, feel free to ask!',
                'role': 'assistant'
            },
            {
                'content':
                f'QUESTION: How many rows are in the {tables[0].lower()} table?',
                'role': 'user'
            },
            {
                'content':
                f'We can run the following code\n```python\nquery = "SELECT COUNT(*) '
                f'FROM {tables[0]};"\nrst = cursor.execute(query).fetchall()\n```',
                'role':
                'assistant'
            },
            {
                'content':
                "exitcode: 0 (execution succeeded)\nCode output: \n{'rst': (" +
                str(n_rows_table0) + ",)}",
                'role':
                'user'
            },
            {
                'content':
                f'ANSWER: There are {n_rows_table0} rows in the {tables[0]} table.',
                'role': 'assistant'
            },
            {
                'content':
                f'QUESTION: What columns are in the {tables[0].lower()} table?',
                'role': 'user'
            },
            {
                'content': 'We can run the function\n'
                '```python\nrst = show_table_schema(cursor, {tables[0]})\n```',
                'role': 'assistant'
            },
            {
                'content':
                "exitcode: 0 (execution succeeded)\nCode output: \n{'rst': " +
                str(table0_info) + "}",
                'role':
                'user'
            },
            {
                'content':
                f'ANSWER: The columns in the {tables[0]} table are: ' +
                ", ".join(column_names(self.cursor, tables[0])),
                'role':
                'assistant'
            },
            {
                'content': "QUESTION: Who are you?",
                'role': 'user'
            },
            {
                'content':
                "ANSWER: Sorry, I don't have enough information to answer your "
                "question because it is not in the database.",
                'role':
                'assistant'
            },
        ]
        self.assistant._oai_messages[self.proxy] = prior

    def generate_sql_reply(self, messages: Optional[List[Dict]],
                           sender: "Agent", config):
        """Generate a reply using OpenAI DALLE call."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        prompt = messages[-1]["content"]

        self.proxy.initiate_chat(self.assistant,
                                 message="QUESTION: " + prompt,
                                 clear_history=False)
        ans = content_str(self.assistant.last_message()["content"])
        ans = ans.replace("ANSWER:", "").strip().rstrip()

        return True, ans


class SQLProxy(UserProxyAgent):

    def __init__(self,
                 sql_file,
                 name,
                 code_execution_config: dict = DEFAULT_SQL_CODE_CONFIG,
                 **kwargs):
        super().__init__(name,
                         code_execution_config=code_execution_config,
                         **kwargs)
        self.sql_file = sql_file
        connection = sqlite3.connect(sql_file)
        self.cursor = connection.cursor()

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
        rst = execute_code_with_cursor(code, self.cursor, **kwargs)
        exitcode = 0 if str(rst).find("Error: ") != 0 else 1

        return exitcode, str(rst), None


############# Helper Functions ##############
# Function to list all tables in the database
def list_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    ans = []
    for table in cursor.fetchall():
        ans.append(table[0])
    return ans


# Function to show the schema of a table
def show_table_schema(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")

    ans = f"Here is the table information for Table: {table_name}\n"
    ans += "-" * 30 + "\n"
    ans += "Column ID | Column Name | Data Type | Not Null Constraint | Default Val "
    ans += "| Primary Key | \n"
    ans += "-" * 30 + "\n"
    for row in cursor.fetchall():
        ans += " | ".join([str(x) for x in row]) + "\n"
    ans += "-" * 30 + "\n\n\n"
    return ans


# Function to execute a query
def execute_query(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()


def find_potential_joins(cursor):
    # Query to get foreign key relationships
    cursor.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    joins = []
    for table, create_statement in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            # Each item in foreign_keys is a tuple like:
            # (id, seq, table, from, to, on_update, on_delete, match)
            from_table = table
            from_column = fk[3]
            to_table = fk[2]
            to_column = fk[4]
            join_info = f"{from_table}.{from_column} -> {to_table}.{to_column}"
            joins.append(join_info)

    return joins


def column_names(cursor, table_name):
    """
    Retrieves the column names for a specified table.

    Args:
    cursor (sqlite3.Cursor): The cursor object connected to the database.
    table_name (str): The name of the table for which to retrieve column names.

    Returns:
    list: A list of column names for the specified table.
    """
    query = f"PRAGMA table_info({table_name});"
    cursor.execute(query)
    # Fetches all rows from the query result, extracts the second column which
    # contains column names
    return [row[1] for row in cursor.fetchall()]


def table_prompt(cursor, question="", model="gpt-4", max_tokens=10e9):
    db_info = ""
    if question:
        db_info = f"## My question\n{question}\n\n\n"

    db_info += "## Table information\n"
    tables = list_tables(cursor)
    db_info += f"There are {len(tables)} tables in the database, which are:"
    for t in tables:
        db_info += "\n- " + t

    joins = find_potential_joins(cursor)
    if joins:
        db_info += "\n\n## Join information\n"
        db_info += "Here are the columns (and their tables) that could be joined:"
        for j in joins:
            db_info += "\n- " + j
        db_info += "\n\n"

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    def n_tokens(x):
        return len(encoding.encode(x))

    if max_tokens is None:
        max_tokens = 10e9

    db_info = db_info
    assert n_tokens(db_info) < max_tokens

    #### First, try full schema info
    schema_info = "\n\n## Schema\n"
    for table in tables:
        table_info = show_table_schema(cursor, table)
        schema_info += table_info

    if n_tokens(schema_info + schema_info) < max_tokens:
        return db_info + schema_info

    #### if not possible, try column names
    column_info = "\n\n## Columns\n"
    for table in tables:
        column_info += f"{table}: {column_names(cursor, table)}\n"

    if n_tokens(column_info + column_info) < max_tokens:
        return db_info + column_info

    alert = "\n\n### Schemas are Skipped for some tables.\n__Note__: "
    alert += "You SHOULD use the `column_names(cursor, table_name)` "
    alert += "function to show columns of a table."
    return db_info + alert


def execute_code_with_cursor(code, cursor, **kwargs) -> Dict:
    locals_dict = {
        "cursor": cursor,
        "show_table_schema": show_table_schema,
        "column_names": column_names
    }

    if 'lang' in kwargs and kwargs['lang'].lower() == "sql":
        print(
            colored(
                "WARNING: raw sql code is received for execution. "
                "We allow it for now, but the LLM should be improved.",
                "yellow"))
        code = f"rst = cursor.execute({code}).fetchall()"

    try:
        exec(code, locals_dict, locals_dict)
    except Exception as e:
        return "Error: " + str(e)
    if "rst" in locals_dict:
        return {"rst": locals_dict["rst"]}
    known_keys = list(locals().keys()) + list(globals().keys())
    return {k: v for k, v in locals_dict.items() if k not in known_keys}
