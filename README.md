# DeepEAG: A deep learning-based hybrid framework for identifying epilepsy-associated genes using a stacking strategy
===========================================================================


[![license](https://img.shields.io/badge/python_-3.8.0_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-1.12.0_-blue)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scipy_-1.5.0_-blue)](https://scipy.org/)
[![license](https://img.shields.io/badge/xgboost_-1.0.2_-blue)](https://xgboost.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/scikit_learn_-1.2.0_-blue)](https://scikit-learn.org/)
[![license](https://img.shields.io/badge/numpy_-1.23.5_-blue)](https://numpy.org/)

As one of the most common neurological diseases in the world, epilepsy can be caused through the deletion or duplication of known epilepsy-associated genes. Therefore, it is crucial to accurately identify epilepsy-associated genes to gain a deeper understanding of the pathogenesis of epilepsy and to develop new therapies. Although a number of wet-lab experimental methods have been proposed, they are reliable but in general time-consuming and labor-intensive. Hence, their practical application in epilepsy-associated genes identification is quite limited. To address the existing limitations, we propose a computational method, called DeepEAG, for the identification of epilepsy-associated genes throughout the human genome. To develop DeepEAG, we first constructed a new and, to our knowledge, the most comprehensive epilepsy genomic benchmark dataset. Afterwards, we investigated four feature encoding algorithms with different perspectives and trained them using well-established traditional classifiers and deep learning classifier, resulting in 13 baseline models. Finally, the stacking strategy is effectively utilized by integrating the predicted output of the optimal baseline models and training with xgboost. In results, the DeepEAG predictor obtained excellent performance on the cross-validation with accuracy and AUC of 0.835 and 0.907, respectively. Overall, DeepEAG showed more stable and accurate predictive performance compared to baseline models, and also outperforms other ensemble strategies, demonstrating the effectiveness of our proposed hybrid framework. In addition, DeepEAG is anticipated to facilitate community-wide efforts to identify putative epilepsy-associated genes and generate new biological insights and testable hypotheses. The schematic diagram of DeepEAG is shown as follows:


![Image text](https://github.com/jiboyalab/multiPPIs/blob/main/IMG/flowchartnew.svg)



# Table of Contents

- [Installation](#installation)
- [Data description](#data-description)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Contacts](#contacts)
- [License](#license)


# Installation

DeepEAG is tested to work under:

```
* Python 3.8.0
* Torch 1.12.0
* Scipy 1.5.0
* Xgboost 1.0.2
* Scikit-learn 1.2.0
* Numpy 1.23.5
* Other basic python toolkits
```
# Data description

| File name  | Description |
| ------------- | ------------- |
| MDCuiMiDisease.csv  | MiRNA-disease associations obtained from HMDD v3.0 database |
| DPDrugBankDrugProtein5.csv  | Drug-protein associations obtained from DrugBank v5.0 database  |
| LMSNPLncMi.csv  | MiRNA-lncRNA associations obtained from lncRNASNP2 database  |
| LDAllLncDisease.csv| LncRNA-disease associations obtained from lncRNASNP2 and LncRNADisease database  |
| DrugDiseaseDrugDisease.csv| Drug-disease associations obtained from CTD:update 2019 database| 
| PDDisGeNETProteinDisease20.csv|  Protein-disease associations obtained from DisGeNET database| 
| MPmiRTarBaseMiProtein5.csv| MiRNA-protein associations obtained from miRTarBase: update 2018 database| 
| LPLncRNA2TargetLncProtein3.csv|  LncRNA-protein associations obtained from LncRNA2Target v2.0 database| 
| PPI.csv| Protein-protein interactions obtained from STRING database for model training and prediction| 
| AllProteinSequence.csv| Original protein sequence| 

# Quick start
To reproduce our results:

## 1，Data preprocessing and construction of multi-source heterogeneous biomolecular networks
```
python ./src/data_process.py --path ./data/  && python ./src/GetNodeEdgeNum.py --save ./data/ && python ./src/GetObjectEdgeNum.py --save ./data/
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **path** | Path of the assciation data among different biomoleculars|
| **save** | Save path of the generated data|



## 2，Calculate the protein attribute features
```
python ./src/ACDencoder.py --path ./data/ --save ./data/
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **path** | Path of the protein sequence data|
| **save** | Save path of the generated data|


## 3，Calculate the protein graph features
```
python -m openne --model deepwalk --dataset ./data/AllNodeEdge.csv --num-paths 10 --path-length 80 --window 10

```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **dataset** | The graph data. |
| **num-paths** | Number of random walks that starts at each node. |
| **path-length** | Length of random walk started at each node. |
| **window** | Window size of skip-gram model.  |



## 4，Training and prediction under 5-fold cross-validation
```
python ./src/training.py && python ./src/roc_plot.py && python ./src/pr_plot.py
```








# Contributing

All authors were involved in the conceptualization of the proposed method. LWX and SLP conceived and supervised
the project. BYJ and HTZ designed the study and developed the approach. BYJ and HTZ implemented and applied the method to microbial data. BYJ and HTZ analyzed the results. LWX and SLP contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.



# Contacts
If you have any questions or comments, please feel free to email: byj@hnu.edu.cn, slpeng@hnu.edu.cn.

# License

[MIT Richard McRichface.](../LICENSE)
