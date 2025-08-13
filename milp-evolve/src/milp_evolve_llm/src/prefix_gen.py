import json
import os
from pathlib import Path

milp_path = Path(os.path.dirname(__file__)) / ".." / "milp_class_gen" 
formulation_path = Path(os.path.dirname(__file__)) / ".." / "milp_formulation_gen"
domain_path = conv_path = topic_path = Path(os.path.dirname(__file__)) / ".." / "milp_conv_gen" 

def gen_milp_code(path=milp_path):
    milp_lists = [(path / f).read_text() for f in os.listdir(path) if f.endswith(".py") and f != "__init__.py" and f != "milp_gen.py"]
    return milp_lists

def get_formulation_methods(path=formulation_path):
    return json.load(open(os.path.join(path, "formulations.json"), "r"))


def gen_domain_prefix(path=domain_path):
    PREFIX_TEMPLATE = "You are a professional from the {company} in the {industry} industry."
    domain_info = json.load(open(os.path.join(path, "domains.json"), "r"))

    for industry_name, industry_info in domain_info.items():
        for company in industry_info["famous_companies"]:
            prompt = PREFIX_TEMPLATE.format(company=company, industry=industry_name)
            yield prompt


def gen_conv_prefix(path=conv_path):
    PREFIX_TEMPLATE = "Two professionals from the {company} are discussing {topic} their industry {industry}."

    milp_topics = json.load(open(os.path.join(path, "milp_topics.json"), "r"))
    domain_info = json.load(open(os.path.join(path, "domains.json"), "r"))
    topics = [x for main_topic, info in milp_topics.items() for x in info["subtopics"]]

    topics = set(topics) # remove duplicates
    for topic in topics:
        for industry_name, industry_info in domain_info.items():
            for company in industry_info["famous_companies"]:
                prompt = PREFIX_TEMPLATE.format(topic=topic, company=company, industry=industry_name)
                yield prompt


def gen_topic_prefix(path=topic_path):
    # we uploaded the zipped prefix_prompts.json.tar.gz file to github to save space
    if not os.path.exists(os.path.join(path, "prefix_prompts.json")):
        if not os.path.exists(os.path.join(path, "prefix_prompts.json.tar.gz")):
            raise FileNotFoundError("prefix_prompts.json and prefix_prompts.json.tar.gz both not found")

        import tarfile
        with tarfile.open(os.path.join(path, "prefix_prompts.json.tar.gz"), "r:gz") as tar:
            tar.extractall(path, filter="data"))
    
    topic_info = json.load(open(os.path.join(path, "prefix_prompts.json"), "r"))  
    topic_info = topic_info[-5000:]

    for topic in topic_info:
        if "#Given Prompt#:" in topic["text"]:
            yield topic["text"].split("#Given Prompt#:")[1].strip()
        else:
            yield topic["text"].strip()
