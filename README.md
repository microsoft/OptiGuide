# OptiGuide: Large Language Models for Supply Chain Optimization

> Supply chain operations traditionally involve a variety of complex decision making problems. Over the last few decades, supply chains greatly benefited from advances in computation, which allowed the transition from manual processing to automation and cost-effective optimization. Nonetheless, business operators still need to spend substantial efforts in explaining and interpreting the optimization outcomes to stakeholders. Motivated by the recent advances in Large Language Models (LLMs), we study how this disruptive technology can help bridge the gap between supply chain automation and human comprehension and trust thereof. We design OptiGuide – a framework that accepts as input queries in plain text, and outputs insights about the underlying optimization outcomes. Our framework does not forgo the state-of-the-art combinatorial optimization technology, but rather leverages it to quantitatively answer what-if scenarios (e.g., how would the cost change if we used supplier B instead of supplier A for a given demand?). Importantly, our design does not require sending proprietary data over to LLMs, which can be a privacy concern in some circumstances. We demonstrate the effectiveness of our framework on a real server placement scenario within Microsoft’s cloud supply chain. Along the way, we develop a general evaluation benchmark, which can be used to evaluate the accuracy of the LLM output in other scenarios.


This repository includes the following material to open source the OptiGuide project:
- Research code base for the OptiGuide framework in the [optiguide/optiguide.py](optiguide/optiguide.py) file.
- A demo notebook of the OptiGuide framework [notebook/optiguide_example.ipynb](notebook/optiguide_example.ipynb)
- Benchmarks (dataset) in [benchmark/](benchmark/) for evaluating language models for supply chain applications
- [ ] Benchmark utils for future evaluation.
- [ ] Create a GitHub release for the benchmark.

## Citation
Please cite the paper if you use this code in your own work:
```latex
@article{li2023large,
  title={Large Language Models for Supply Chain Optimization},
  author={Li, Beibin and Mellou, Konstantina and Zhang, Bo and Pathuri, Jeevan and Menache, Ishai},
  journal={arXiv preprint arXiv:2307.03875},
  year={2023}
}
```

## Setup for the OptiGuide Code

### Installation
Once Python is installed, just run `pip install optiguide` to install OptiGuide.

Here is the [Pypi page](https://pypi.org/project/OptiGuide/) for more information.

### Tutorial Option 1: Run from Colab Directly
1. Check the Colab Notebook:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13eMJOcy79mhyEUYicSmbEm63KS7mzi33?usp=sharing)
2. Create a `OAI_CONFIG_LIST` file inside the Colab environment, so that it can connect to the OpenAI service.
  ```json
  [
      {
          "model": "gpt-4",
          "api_key": "Your API Key Comes Here",
          "api_type": "azure",
          "api_base": "The API base URL",
          "api_version": "2023-03-15-preview"
      }
  ]
  ```
1. Now, you can run the Colab Code.

### Tutorial Option 2: Run Locally
1. Install Python and Python packages `pip install optiguide`
2. Install and setup "Gurobi Optimizer" from [Gurobi's official website](https://www.gurobi.com/downloads/gurobi-software/), and you can get a trial license for academic use. Make sure the license is setup correctly.
3. Run OptiGuide example, you can setup the OpenAI API (key and secret) and then test the code with Jupyter notebook [example](`notebook/optiguide_example.ipynb`).
  Example `notebook/OAI_CONFIG_LIST` file
  ```json
  [
      {
          "model": "gpt-4",
          "api_key": "Your API Key Comes Here",
          "api_type": "azure",
          "api_base": "The API base URL",
          "api_version": "2023-03-15-preview"
      }
  ]
  ```

### For Benchmark Evaluation
1. Install Python and Python packages `pip install optiguide gurobipy requests beautifulsoup4`
2. Download the Gurobi examples `python download.py`


## Implementation
### OptiGuide Implementation
We re-implemented OptiGuide for research purposes using [autogen](https://github.com/microsoft/autogen), a framework that enables the development of LLM applications with multiple agents that can converse with each other to solve tasks.

### Evaluation Implementation
We simplified the evaluation process by using Gurobi and Gurobi examples, as detailed below.

## Benchmark Dataset
We have gathered several Python applications from Gurobi's [example code](https://www.gurobi.com/documentation/current/examples/python_examples.html). These applications are copyrighted by Gurobi. The specific examples we are using include:
- diet.py
- facility.py
- netflow.py
- tsp.py
- workforce1.py

Subsequently, we manually crafted questions and provided the ground truth answer code for each question. These labeled questions and answers can be found in the [benchmark/macro](benchmark/macro/) folder.

To download the dataset from Gurobi and automatically edit the source code, execute `python download.py`. The source code for the benchmarked applications will then be saved in the [benchmark/application](benchmark/application/) folder.

Ensure you have the necessary packages installed using:
`pip install requests beautifulsoup4`

Additionally, we developed a [Coffee](benchmark/application/coffee.py) application to showcase the OptiGuide framework as described in our paper.


**We acknowledge that both human and machine may contain errors while generating this benchmark. If you identify any inaccuracies, feel free to open an issue or submit a pull request for correction.**


## Responsible AI Considerations

The integration of Large Language Models (LLMs) into supply chain management has improved efficiency and offered human-like understanding. However, it's crucial to actively consider the ethical and responsible aspects of AI to ensure its application is both beneficial and free from unexpected consequences.

Since we use publicly available language models, our framework naturally adopts both the strengths and weaknesses of the underlying model. Aware of these challenges, we've implemented a safeguard mechanism to ensure the fairness and safety of the results. Yet, despite these safeguards, the complex nature of AI and its inherent flaws can give rise to issues like model biases, adversarial attacks, and other anomalies.

Researchers must critically assess and interpret these AI models, maintaining a comprehensive understanding of both the advantages and challenges of LLMs. By responsibly utilizing the capabilities of these models, we can achieve groundbreaking progress in supply chain practices and optimization, all while upholding ethical and unbiased principles.

## Prohibitions
Due to the specialized nature of this repository, which includes both benchmark data and source code designed for evaluating data-oriented models and pipelines, we expressly prohibit the following:

1. Scraping the content of this repository for the purpose of training machine learning models, deep learning architectures, data science algorithms, large language models, or any other data-driven computational models.  Utilizing this repository's content for such purposes could introduce bias and invalidate the evaluation metrics of the trained models. You are, however, permitted to use this repository for the evaluation of models and pipelines.
2. Violating the licensing terms in this repository.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
