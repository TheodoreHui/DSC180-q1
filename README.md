# ConWea Baseline Replication Project

## Overview

This project focuses on utilizing Natural Language Processing (NLP) techniques, specifically TF-IDF and Word2Vec, to classify documents into relevant categories as a replication of the baselines of "Contextualized Weak Supervision for Text Classification" (Dheeraj Mekala, Jingbo Shang)

https://aclanthology.org/2020.acl-main.30/

### Accessing Data
Training data that was used in the original project is accessesible in this repo. Assuming the file structure remains unchanged, the test scripts should be pathed toward them already. However, new data can be used, given they following the formatting requirements in the documentation.

### Dependencies
All necessary libraries should be in the requirements.txt, which can be executed via:
pip install -r requirements.txt

### Reproducing
To make these techniques more modular, they have been written as classes. Each source .py file should contain the documentation necessary for utilizing these models, which should be similar to existing machine learning paradigms. Additionally, test notebooks have been provided with results from the original paper as well as the results of the recreated techniques. 
