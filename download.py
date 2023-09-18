"""Download and prepare evaluation examples and benchmarks.

Download example code from Gurobi, extract the source code, and prepare it for
the OptiGuide.

Acknowledgement:
    This code is inspired by and may contain snippets or structures based on
    example codes provided in Gurobi's official documentation.
    Please refer to Gurobi's documentation for further examples and details.

Dependencies:
    - requests
    - BeautifulSoup
"""

import os
import re
import time
from typing import Union

import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, RequestException, Timeout


def fetch_and_extract_content(url: str,
                              timeout_seconds: int = 10) -> Union[str, None]:
    """
    Fetch the page content from a given URL and extract specific content.

    Parameters:
        url (str): The URL of the page to fetch.
        timeout_seconds (int): Timeout in seconds for the request.
        Default is 10 seconds.

    Returns:
        Union[str, None]: The extracted content if found, otherwise None.

    Raises:
        HTTPError: For HTTP errors.
        Timeout: For request timeouts.
        RequestException: For general request exceptions.
    """

    try:
        # Fetch the page content with a timeout
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract content within <PRE CLASS="prettyprint"> and </PRE>
        content = soup.find('pre', class_='prettyprint')

        return content.text if content else None

    except Timeout:
        print(f"Request to {url} timed out.")
        return None
    except HTTPError as e:
        print(f"HTTP Error occurred: {e}")
        return None
    except RequestException as e:
        print(f"An error occurred while fetching {url}: {e}")
        return None


def handle_source_code(code: str) -> str:
    """
    Process a given source code string to insert specific comments and modify
    lines based on certain conditions.

    The function performs the following modifications:
    1. Ignores lines starting with '#'.
    2. Inserts a comment "# OPTIGUIDE DATA CODE GOES HERE" before lines
    containing "gp.Model(" and appends "model = m" after the line.
    3. Inserts a comment "# OPTIGUIDE CONSTRAINT CODE GOES HERE" before lines
    containing ".optimize(".

    Parameters:
    - code (str): The source code string to be processed.

    Returns:
    - str: The processed source code with inserted comments and modifications.

    Example:
    >>> code = '''
    ... # This is a comment
    ... m = gp.Model("test")
    ... result = m.optimize()
    ... '''
    >>> handle_source_code(code)
    '''
    # OPTIGUIDE DATA CODE GOES HERE
    m = gp.Model("test")
    model = m
    # OPTIGUIDE CONSTRAINT CODE GOES HERE
    result = m.optimize()
    '''
    """
    lines = code.splitlines()

    final_lines = []
    for line in lines:
        if line.strip().startswith('#'):
            continue

        if line.find("gp.Model(") >= 0:
            # Create model
            final_lines += [
                "# OPTIGUIDE DATA CODE GOES HERE", line, "model = m"
            ]
            continue
        elif line.find(".optimize(") >= 0:
            # Before adding the constraints
            final_lines.append(line)
            final_lines.append("\n# OPTIGUIDE CONSTRAINT CODE GOES HERE\n")
            final_lines.append("m.update()")

        if line.find("sys.exit(") >= 0:
            line = re.sub(r"sys\.exit\((\d*)\)", r"quit()", line)

        final_lines.append(line)

    return "\n".join(final_lines)


def special_handle(url: str, code: str) -> str:
    """Special handling of some code.

    - For netflow.py:
    Add the `original_solution` variable into the  file, which will
    be used in the benchmark.


    - For tsp.py
    Change the code to use a fixed number of points.

    - For workforce1.py
    Reduce the number of required shifts by half, so that we would have
    meaningful analysis in the benchmark. We also remove the bottom code, which
    might exit the program unexpectedly.
    """
    if url.find("netflow") >= 0:

        extra_code = """
# For retrival in question markdown, etc.
original_solution = [
    ("Pencils", "Detroit", "Boston"),
    ("Pencils", "Denver", "New York"),
    ("Pencils", "Denver", "Seattle"),
    ("Pens", "Detroit", "Boston"),
    ("Pens", "Detroit", "New York"),
    ("Pens", "Denver", "Boston"),
    ("Pens", "Denver", "Seattle"),
]

"""
        return extra_code + code

    if url.find("tsp") >= 0:
        code = re.sub(r"if len\(sys.argv\) < 2.+\(sys.argv\[1\]\)",
                      "\nn = 10\n\n",
                      code,
                      flags=re.DOTALL)

        code += "\nm._num_points = n\n"

        return code

    if url.find("workforce1") >= 0:
        code = code.replace(
            "# OPTIGUIDE DATA CODE GOES HERE", """
shiftRequirements = {k: max(1, int(v / 2)) for k, v in
                      shiftRequirements.items()}
# OPTIGUIDE DATA CODE GOES HERE
availability = gp.tuplelist(list(set(availability)))""")

        idx = code.find("status = m.Status")
        return code[:idx]

    return code


# URLS
urls = [
    'https://www.gurobi.com/documentation/10.0/examples/diet_py.html',
    'https://www.gurobi.com/documentation/10.0/examples/facility_py.html',
    'https://www.gurobi.com/documentation/10.0/examples/netflow_py.html',
    'https://www.gurobi.com/documentation/10.0/examples/tsp_py.html',
    'https://www.gurobi.com/documentation/10.0/examples/workforce1_py.html'
]

if __name__ == "__main__":
    # Example
    for url in urls:
        code = fetch_and_extract_content(url)
        edited_code = handle_source_code(code)
        edited_code = special_handle(url, edited_code)
        outname = url.split('/')[-1].replace('_py.html', '.py')
        outname = os.path.join("benchmark/application", outname)
        open(outname, 'w').write(edited_code)
        print(outname, "written...")
        time.sleep(1)
