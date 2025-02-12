---
language: English
license: cdla-2.0
multilinguality: monolingual
size_categories:
  - 100K<n<1M
source_datasets: original
task_categories:
  - multi-task
  - regression
  - reinforcement-learning
  - other
task_ids:
  - integrality-gap-prediction
  - learning-to-branch
  - language-milp-alignment
---

# Dataset Card for MILP-Evolve

## Table of Contents
- [Dataset Card for MILP-Evolve](#dataset-card-for-milp-evolve)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)
  - [Trademarks](#trademarks)

## Dataset Description

- **Homepage:** [The OptiGuide Project](https://www.microsoft.com/en-us/research/project/optiguide-genai-for-supply-chain-optimization/?msockid=1a1ccce4197d663e1c2bdd4318e1678d)
- **Repository:** [MILP-Evolve](https://github.com/microsoft/MILP-Evolve)
- **Paper:** [arXiv]([arXiv](https://arxiv.org/abs/2410.08288))
- **Leaderboard:** Beibin Li, Ishai Menache, Sirui Li, Janardhan Kulkarni, Cathy Wu
- **Point of Contact:**
  - [beibin.li@microsoft.com](mailto:beibin.li@microsoft.com)
  - Sirui Li

### Dataset Summary

MILP-Evolve is a large-scale dataset of Mixed Integer Linear Programming (MILP) problem classes and instances. It is generated using an LLM-based evolutionary framework capable of producing a diverse set of MILP classes with unlimited instances. The dataset is designed to facilitate research in developing foundation models for MILP that generalize across problem classes. It supports multiple learning tasks, including integrality gap prediction, learning to branch, and aligning MILP instances with natural language descriptions.

### Supported Tasks and Leaderboards

- **Integrality Gap Prediction:** This task involves predicting the integrality gap of MILP instances, which measures the difference between the optimal integer solution and the linear relaxation. Success is typically measured by a low mean squared error or high correlation between predicted and actual gaps. Models like regression-based neural networks can be trained using this dataset.

- **Learning to Branch:** The dataset can be used to train models that learn effective branching strategies within MILP solvers. Performance is measured by metrics such as reduced solve time, smaller branch-and-bound trees, or fewer nodes explored. Reinforcement learning models or imitation learning approaches are commonly used for this task.

- **Language-MILP Alignment:** This new task involves aligning MILP instances with their natural language descriptions. Success is measured by retrieval accuracy or alignment scores. Models like cross-modal transformers or contrastive learning frameworks can be applied.

### Languages

The dataset is primarily in English. It includes Python code for MILP formulations, natural language descriptions in English, and standard MILP file formats like MPS.

## Dataset Structure

### Data Instances

An example of a data instance for integrality gap prediction:

```json
{
  "milp_class": "ConferenceRoomScheduling",
  "instance_id": "CRS_001",
  "num_variables": 500,
  "num_constraints": 300,
  "density": 0.05,
  "integrality_gap": 0.12
}
```

An example for learning to branch:

```json
{
  "milp_instance": "SetCover_123",
  "branching_decisions": [
    {"node": 1, "variable": "x_5", "decision": "branch_up"},
    {"node": 2, "variable": "x_12", "decision": "branch_down"}
    // ...
  ],
  "solve_time": 150.5
}
```

An example for language-MILP alignment:

```json
{
  "milp_instance": "ResourceAllocation_456",
  "description": "Optimize the allocation of resources to maximize profit while adhering to budget constraints.",
  "graph_features": {
    "nodes": [...],
    "edges": [...],
    "node_features": [...],
    "edge_features": [...]
  }
}
```

### Data Fields

- **milp_class:** String identifier for the MILP class.
- **instance_id:** Unique identifier for each MILP instance.
- **num_variables:** Integer count of variables in the instance.
- **num_constraints:** Integer count of constraints in the instance.
- **density:** Float representing the density of the constraint matrix.
- **integrality_gap:** Float value of the integrality gap (for integrality gap prediction task).
- **branching_decisions:** List of branching decisions taken during the solve (for learning to branch task).
- **solve_time:** Float representing the time taken to solve the instance.
- **description:** Natural language description of the MILP instance (for language-MILP alignment task).
- **graph_features:** Graph representation of the MILP instance, including nodes, edges, and associated features.

### Data Splits

For **Integrality Gap Prediction**:

- **Training Set:** 38,256 instances from 643 classes.
- **Validation Set:** 9,564 instances from the same classes as training.
- **Test Set:** 11,584 instances from 157 unseen classes.

For **Learning to Branch**:

- **Training Set:** Data from 579 classes, totaling 26,502 instances.
- **Validation Set:** Data from 59 classes, totaling 512 instances.
- **Test Set:** Data from 162 classes, totaling 4,756 instances.

For **Language-MILP Alignment**:

- **Training Set:** Instances from 80% of the classes.
- **Test Set:** Instances from the remaining 20% of the classes.

## Dataset Creation

### Curation Rationale

The dataset was created to overcome the limitations of existing MILP datasets, which often lack diversity and volume, hindering the generalization of deep learning models across different problem classes. MILP-Evolve aims to provide a diverse and extensive dataset to facilitate the development of foundation models in MILP.

### Source Data

#### Initial Data Collection and Normalization

MILP-Evolve uses an LLM-based evolutionary framework to generate MILP code iteratively. Starting from 8 seed classes from previous literature, the framework employs OpenAI's GPT-4 to generate new MILP classes by applying various transformations like addition, mutation, and crossover. Each generated code is then subjected to parameter tuning and filtering to ensure computational feasibility and diversity.

#### Who are the source language producers?

The source data is machine-generated by the MILP-Evolve framework using GPT-4. The initial seed classes are standard MILP problems from established literature, reformatted into a modular code structure.

### Annotations

#### Annotation process

Annotations such as integrality gaps and branching decisions are generated automatically. For integrality gaps, instances are solved using MILP solvers with specified time limits, and gaps are calculated based on the optimal solutions. For learning to branch, data is collected by solving instances and recording branching decisions, sometimes employing expert strategies like Strong Branching.

#### Who are the annotators?

Annotations are produced by computational processes and solvers without human intervention.

### Personal and Sensitive Information

The dataset does not contain personal or sensitive information. All data is synthetic and generated for research purposes.

## Considerations for Using the Data

### Social Impact of Dataset

MILP-Evolve has the potential to significantly advance optimization and machine learning research. By enabling models that generalize across a wide range of MILP problems, it can lead to more efficient solutions in industries like logistics, supply chain, healthcare, and environmental planning. This can result in cost savings, improved resource utilization, and better decision-making processes.

This dataset is being released to facilitate further machine learning research and not for any real-world application. Users are responsible for developing, testing, and validating any models trained on this dataset before any implementation in the real world.

### Discussion of Biases

While efforts were made to ensure diversity, the dataset may still reflect biases inherent in the LLM's training data or the initial seed classes. Certain problem types or formulations might be overrepresented, and users should be cautious when generalizing results.

### Other Known Limitations

- **Generative Errors:** As the MILP classes are generated by an LLM, there might be syntactic or logical errors that passed the filtering process.
- **Computational Feasibility:** Some instances might be trivial or extremely hard to solve despite filtering for problem size and solve time.
- **Representation Gaps:** Despite the dataset's size, it may not cover all real-world MILP applications.

## Additional Information

### Dataset Curators

The dataset was curated by the research team behind the MILP-Evolve framework. Specific names and affiliations will be provided upon publication.

### Licensing Information

This dataset is licensed under the [CDLA-2.0](https://cdla.dev/permissive-2-0/).

### Citation Information

Please cite the following paper when using the dataset:

```latex
@article{li2024towards,
  author    = {Li, Sirui and Kulkarni, Janardhan and Wu, Cathy and Menache, Ishai and Li, Beibin},
  title     = {Towards Foundation Models for Mixed Integer Linear Programming},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```

### Contributions

Thanks to the entire MILP-Evolve team for their efforts in creating and releasing this dataset.

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
