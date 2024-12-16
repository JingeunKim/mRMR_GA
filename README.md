# A Study on the Effectiveness of Feature Selection considering mRMR across Different Datasets

This repository contains the code for the research paper titled "A Study on the Effectiveness of Feature Selection considering mRMR across Different Datasets".

# Abstract
Rapid advancements in computer technology have led to the generation of large amounts of data, which have degraded machine learning performance due to high dimensionality and irrelevant data. Conventional feature selection methods select a feature subset that satisfies minimum redundancy and maximum relevance (mRMR). In this study, we discuss the performance of the feature selection method considering mRMR and the method that considers only MR using GAs (mRMR_GA and MR_GA) for various datasets with different characteristics and different numbers of features to explore the efficacy of minimum redundancy (mR). First, we analyze the performance of mRMR_GA and MR_GA, comparing them with other feature selection methods to explore the impact of mR. Second, we investigate the effectiveness of mRMR_GA and MR_GA using various datasets with diverse dimensions from 9 to 17,536. For our experiments, we used four high-dimensional datasets and five low-dimensional datasets. After applying the feature selection methods, we classified the dataset using extreme gradient boosting (XGBoost), deep neural network (DNN), decision tree (DT), support vector machine (SVM), and k-nearest neighbor (KNN). We evaluated the models based on accuracy, F1 score, and time cost. The experiments demonstrated that, the higher the dimension, the more effectively mRMR_GA improved the performance of machine learning over MR_GA. Additionally, we found that the effectiveness was more pronounced when the number of selected features is relatively small compared to the total number of features.


# Datasets
Datasets after applying various feature selection methods: [Google Drive](https://drive.google.com/drive/folders/11mj-tb_E3LN0hfbmwqyOnHv03AxCxnCN?usp=drive_link)


## Usage 
```
python ../main.py\
   --filename <dataset_name>\
  --human_number <number_of_humans>\
  --gene_number <number_of_genes>\
  --generation_num <number_of_generations>\
  --weight <weight_value>\
  --equation <equation_type>\
  --abs <abs_value>\
  --method <method_type>\
  --mode <mode_type>
```



## Citation
If you find this useful, please cite the following paper:
```
TBA
```
