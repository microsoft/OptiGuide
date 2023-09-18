# OptiGuide: Large Language Models for Supply Chain Optimization

> Supply chain operations traditionally involve a variety of complex decision making problems. Over the last few decades, supply chains greatly benefited from advances in computation, which allowed the transition from manual processing to automation and cost-effective optimization. Nonetheless, business operators still need to spend substantial efforts in explaining and interpreting the optimization outcomes to stakeholders. Motivated by the recent advances in Large Language Models (LLMs), we study how this disruptive technology can help bridge the gap between supply chain automation and human comprehension and trust thereof. We design OptiGuide – a framework that accepts as input queries in plain text, and outputs insights about the underlying optimization outcomes. Our framework does not forgo the state-of-the-art combinatorial optimization technology, but rather leverages it to quantitatively answer what-if scenarios (e.g., how would the cost change if we used supplier B instead of supplier A for a given demand?). Importantly, our design does not require sending proprietary data over to LLMs, which can be a privacy concern in some circumstances. We demonstrate the effectiveness of our framework on a real server placement scenario within Microsoft’s cloud supply chain. Along the way, we develop a general evaluation benchmark, which can be used to evaluate the accuracy of the LLM output in other scenarios.


This repository will include the following material to open source the OptiGuide project:
- [x] Research code base for the OptiGuide framework
- [x] A demo notebook of the OptiGuide framework
- [x] Benchmarks (dataset) for evaluating language models for supply chain applications
- [] version control the research code with FLAML library


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
