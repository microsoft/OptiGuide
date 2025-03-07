import os
import re

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()

install_requires = [
    "openai>=1.3", "diskcache", "termcolor", "autogen-agentchat~=0.2.0", "eventlet",
    "gurobipy"
]

__version__ = re.findall("\"(.+)\"", open("what-if/optiguide/version.py").read())[0]

setuptools.setup(
    name="OptiGuide",
    version=__version__,
    author="Beibin Li",
    author_email="beibin.li@microsoft.com",
    description="GenAI for Decision Making and Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/optiguide",
    packages=setuptools.find_packages(include=["what-if/optiguide*"]),
    package_data={
        "optiguide.default": ["*/*.json"],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "test": [
            "pytest>=6.1.1",
            "pre-commit",
            "nbconvert",
            "nbformat",
            "ipykernel",
            "pydantic==1.10.9",
            "sympy",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
