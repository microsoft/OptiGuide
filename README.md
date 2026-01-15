# OptiGuide: GenAI for Optimization and Decision Intelligence


## Projects

We have several projects in this repository for decision making and optimization, and please cite the paper if you use this code in your own work.


### OptiGuide for Supply Chain Optimization with What-if Analysis
The [what-if](what-if/) folder includes the following material to open source the paper and what-if analysis for the OptiGuide project:

```latex
@article{li2023large,
  title={Large Language Models for Supply Chain Optimization},
  author={Li, Beibin and Mellou, Konstantina and Zhang, Bo and Pathuri, Jeevan and Menache, Ishai},
  journal={arXiv preprint arXiv:2307.03875},
  year={2023}
}
```

### Towards Foundation Models for Mixed Integer Linear Programming
The [milp-evolve](milp-evolve/) folder includes the following material to open source the paper, MILP-Evolve code, and data:

```latex
@article{li2024towards,
  author    = {Li, Sirui and Kulkarni, Janardhan and Wu, Cathy and Menache, Ishai and Li, Beibin},
  title     = {Towards Foundation Models for Mixed Integer Linear Programming},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```

### OptiMind: Teaching LLMs to Think Like Optimization Experts
The [optimind](optimind/) folder includes the following material to open source the paper, evaluation code, and test data for the OptiMind project: 

```
@article{zhang2025optimind,
  title={OptiMind: Teaching LLMs to Think Like Optimization Experts},
  author={Zhang, Xinzhi and Chen, Zeyi and Zope, Humishka and Barbalho, Hugo and Mellou, Konstantina and Molinaro, Marco and Kulkarni, Janardhan and Menache, Ishai and Li, Sirui},
  journal={arXiv preprint arXiv:2509.22979},
  year={2025}
}
```


## Responsible AI Considerations  

The integration of GenAI into decision intelligence and optimization has provided human-like reasoning in complex problem-solving. However, it is essential to actively consider the ethical and responsible aspects of AI to ensure its application remains both beneficial and free from unintended consequences.  

Since we utilize publicly available language models, our framework inherently inherits both the strengths and limitations of these models. Recognizing these challenges, we have implemented safeguard mechanisms to ensure fairness, robustness, and safety in decision-making. Nevertheless, due to the inherent complexities of AI, issues such as model biases, adversarial vulnerabilities, and unexpected behaviors can still arise.  

Researchers and practitioners must critically assess and interpret these AI-driven solutions, maintaining a deep understanding of both the advantages and limitations of LLMs. By responsibly leveraging these models, we can drive significant advancements in decision intelligence and optimization while upholding ethical standards and mitigating potential risks.

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
