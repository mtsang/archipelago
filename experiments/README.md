# Reproducibility

Almost all experiments are run through Jupyter notebooks.

If you haven't done so already, please setup experiments with the following steps:
1. Run the following command to setup experiments
```bash
sh ../setup_experiments.sh
```
2. Download the ImageNet '14 test set from [here](http://www.image-net.org/challenges/LSVRC/2014/) and place it in ```../downloads/imagenet14/```.

## 1. ArchDetect

Navigate to the ArchDetect experiment folder:
```bash
cd 1.\ archdetect/
```

### Table 1b

Method  | F1 | F2 | F3 | F4 
------------- | ------------- | ------------- | ------------- | -------------
Two-way ANOVA  | 1.0  | 0.51  | 0.51  | 0.55
Integrated Hessians  | 1.0  | N/A  | N/A  | N/A
Neural Interaction Detection  | 0.94  | 0.54  | 0.54  | 0.56
Shapley Interaction Index  | 1.0  | 0.50  | 0.50  | 0.51
Shapley Taylor Interaction Detection  | 1.0  | 0.55  | 0.78  | 0.55
ArchDetect  | 1.0  | 1.0  | 1.0  | 1.0


To reproduce these experiments, please use ```1. synthetic_performance.ipynb```.

### Figure 3
<p align="left">
<img src="1. archdetect/redundancy.png" width="350">
</p>

To run redundancy experiments, please use  ```2. redundancy_bert.ipynb``` and ```2. redundancy_resnet.ipynb```. Plotting can be done in ```2.1. redundancy_analysis_plotting.ipynb```.

## 2. ArchAttribute

Navigate to the ArchAttribute experiment folder:
```bash
cd 2.\ archattribute/
```

### Table 2


Method  | Word Correlation | Phrase Correlation | Segment AUC
------------- | ------------- | ------------- | ------------- 
Difference  | 0.333  | 0.639  | 0.705
Integrated Gradients (IG) | 0.473  | 0.737  | 0.786
Integrated Hessians  (IH) | N/A  | 0.128  | N/A
Model-Agnostic Hierarchical Explanations (MAHE) | 0.570  | 0.702  | 0.712
Shapley Interaction Index (SI) | 0.160  | -0.018 | 0.530
Shapley Taylor Interaction Index (STI) | 0.657  | 0.286  | 0.626
Sampling Contextual Decomposition (SCD) | 0.622  | 0.742  | N/A
Sampling Occlusion (SOC) | 0.670  | 0.794  | N/A
ArchAttribute | 0.745  | 0.836  | 0.919

To reproduce these experiments, please use ```text_correlation.ipynb``` and ```segment_auc.ipynb```. Both Word Correlation and Phrase Correlation are evaluated in ```text_correlation```.

To run MAHE, use the correponding python scripts in ```parallel_mahe/```.

To actually compute correlation and AUC scores, you can use the notebooks in ```analysis/```.

