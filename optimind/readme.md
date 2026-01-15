# OptiMind: Teaching LLMs to Think Like Optimization Experts

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.22979-b31b1b.svg)](https://arxiv.org/abs/2509.22979)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OptiMind--SFT-blue)](https://huggingface.co/microsoft/OptiMind-SFT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://mit-license.org/)

</div>

This repository contains the official evaluation code and cleaned datasets for the paper **"OptiMind: Teaching LLMs to Think Like Optimization Experts"**.

OptiMind is a comprehensive framework designed to enhance Large Language Models (LLMs) for formulating Mixed-Integer Linear Programming (MILP) problems. By systematically integrating optimization domain expertise into both training (via data cleaning) and inference (via class-specific error hints), OptiMind significantly improves formulation accuracy over base models and other open-source baselines.

## Overview

The repository provides:
1.  **Evaluation Scripts**: The exact code used in our paper to evaluate models using single-turn generation, majority voting, and multi-turn self-correction with solver feedback.
2.  **Cleaned Benchmarks**: Expert-verified versions of IndustryOR, Mamo-Complex, and OptMATH, where we fixed missing data, ambiguities, and incorrect ground truths.


## Usage

### 1. Install Enviornment
We recommend using `uv` for fast package management. Please follow the steps below to set up the environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .sglang --python 3.12 --seed
source .sglang/bin/activate
uv pip install "sglang" --prerelease=allow
uv pip install pandas numpy matplotlib json_repair nest_asyncio gurobipy hf_transfer azure-identity pyarrow termcolor
```

**Note:** Make sure you have a valid Gurobi license to execute the solver-based evaluation and self-correction loops.

### 2. Model
Our fine-tuned model is available on Hugging Face: [microsoft/OptiMind-SFT](https://huggingface.co/microsoft/OptiMind-SFT).

### 3. Running Evaluations
First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/microsoft/OptiGuide.git
cd OptiMind
```

We provide shell scripts to automate the evaluation pipeline, which includes running experiments across multiple random seeds, temperatures, and majority-voting configurations. Before running the evaluation scripts, you must update the file paths in `submit_eval.sh` and `submit_eval_gptoss.sh` to match your local environment.

For `gpt-oss-20b` and its fine-tuned variants (including [our fine-tuned model](https://huggingface.co/microsoft/OptiMind-SFT)), use `submit_eval_gptoss.sh`:
```
bash submit_eval_gptoss.sh
```
For other open-source models (e.g., Qwen, Llama), use `submit_eval.sh` that excludes the  `--gpt-oss` and `--reasoning` flags.
```
bash submit_eval.sh
```
The evaluation result will be automatically saved in `eval_results/<benchmark_name>/<model_name_and_run_configurations>`. To aggregate the mean and stand deviation across the runs, run
```
python calculate_avg_results.py <path_to_the_run>
```



## Data and Benchmarks
We provide our rigorously cleaned versions of three challenging benchmarks in the `/data` folder. These files are provided in CSV format with question and answer columns.


- `data/optimind_cleaned_classified_industryor.csv`: Cleaned and classified IndustryOR benchmark.
- `data/optimind_cleaned_classified_mamo_complex.csv`: Cleaned and classified Mamo-Complex benchmark.
- `data/optimind_cleaned_classified_optmath.csv`: Cleaned and classified OptMATH benchmark.
- `data/hints_subset.csv`: The dictionary mapping problem classes to specific error summaries and hints (currently only a small illustrative subset is included).

### Dataset Cleaning Comparisons

In the `data/comparison/` folder, we provide detailed HTML summary tables that transparently document every modification made to the original benchmarks. These files allow you to align the original instances with our cleaned versions and verify the specific fixes (e.g., addressing missing parameters, ambiguity, or wrong ground truths).

- `data/comparison/industryOR_original_vs_ours.html`: Comparison table aligning the original [IndustryOR](https://huggingface.co/datasets/CardinalOperations/IndustryOR) instances with our cleaned versions.
- `data/comparison/OptMATH_original_vs_ours.html`: Comparison table aligning the [OptMATH](https://github.com/optsuite/OptMATH/blob/main/benchmark/OptMATH_Bench.json) instances with our cleaned versions.
- `data/comparison/compare_SIRL_Ours.html`: A comparison between our cleaned IndustryOR set and the cleaned version from [SIRL](https://github.com/Cardinal-Operations/SIRL/blob/main/test_data/IndustryOR_fixedV2.json) (identifying residual issues in the latter).

Each row in these HTML tables contains the Problem Index, Original Problem, Original Answer, Updated Problem, Updated Answer, and a "How did we fix it" description.


## License 
This project is licensed under the [MIT Licence](https://mit-license.org/).

## Citation
If you find our work, code, or datasets useful, please cite our paper:
```
@article{zhang2025optimind,
  title={OptiMind: Teaching LLMs to Think Like Optimization Experts},
  author={Zhang, Xinzhi and Chen, Zeyi and Zope, Humishka and Barbalho, Hugo and Mellou, Konstantina and Molinaro, Marco and Kulkarni, Janardhan and Menache, Ishai and Li, Sirui},
  journal={arXiv preprint arXiv:2509.22979},
  year={2025}
}
```
