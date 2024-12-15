# Open-set Cross-network Node Classification via Unknown-excluded Adversarial Graph Domain Alignment (UAGA)
This repository contains the author's implementation in PyTorch for the paper "Open-set Cross-network Node Classification via Unknown-excluded Adversarial Graph Domain Alignment".
# Environment Requirement
The experiments were conducted on a single Tesla A40 GPU with 48GB memory. The required packages are as follows:
- python==3.9.13
- torch==1.7.1
- numpy==1.21.5
- scipy==1.9.1
- scikit-learn==1.0.2
- dgl==0.7.2
# Datasets
data/ contains the 3 datasets used in our paper, i.e., Citation-v1, DBLP-v4, and ACM-v8.

Each ".mat" file stores a network dataset, where

the variable "adjacency_matrix" represents an adjacency matrix,

the variable "features" represents a node attribute matrix,

the variable "labels" represents a node label matrix.
# Code
"models.py" is the UAGA model.

"train.py" is the cross-network node classification over all 6 tasks.
# Please cite our paper as:
Xiao Shen, Zhihao Chen, Shirui Pan, Shuang Zhou, Laurence T. Yang, and Xi Zhou. Open-set Cross-network Node Classification via Unknown-excluded Adversarial Graph Domain Alignment. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), 2025.