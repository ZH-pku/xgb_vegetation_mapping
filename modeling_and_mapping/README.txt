#Program: High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features

Please CITE the following article when using the codes: 

H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing. (submitted)

Contacts

Heng Zhang: heng.zhang@pku.edu.cn; hengzhang.zhh@gmail.com
Anwar Eziz: anwareziz@pku.edu.cn

Summary and Instructions

Files in this directory, as a demo of vegetation mapping, containing python codes and formatted New Zealand (NZ) data, belongs to Modeling & Mapping part in the workflow (Figure 1). In the 'codes' folder, there are 5 python files and a 'src' folder saving all used mapping functions. Before using them, please make sure that the required modules or packages have been correctly installed. The required modules are numpy, pandas, imbalanced-learn, scikit-learn, gdal, and xgboost. The installation instructions could be found in Google with ease. Besides, we recommend to use Anaconda3 python in the program (https://www.anaconda.com/distribution/).

The Modeling & Mapping part starts with main_feature_selection.py, which follows the flow chart of revised Improved Forward Floating Selection (Figure 4). The feature selectio result should be formatted and moved to 'data' folder (use Variable_ID_all.csv in 'data' folder as template). Then, main_multiclass_single.py (single multiclass XGBoost model) or main_multiclass_bagging.py (multiclass XGBoost bagging ensemble model) may be used to map distribution of vegetation with previously selected variables. For vegetation data with hierarchical vegetation classification system (HVCS), the base map layer should be computed firstly. And, main_splitting_mapping.py and main_merging_mapping.py may be executed according to HVCS and the base map layer. Detailed explaination could be seen (as annotation) in the python files. 

The 'data' folder presents a set of formatted NZ data (csv format). This could be helpful in preparing your own data. NZ_vegedata_train.csv is the train set, while NZ_vegedata_test.csv is the test set. Variable_ID_all.csv is variable name list. VegeTypeNames_Habitat.csv is vegetation class name list. To prepare the datasets, please make sure that all variable names are the same in the train, test sets and raster folder (e.g. clim01.tif). The values of variables in the csv files should be equal to the directly extracted values from the rasters (e.g. using ArcGIS software). Note that, all raster files stored in the 'data' folder should have the same resolution and boundary.

Disclaimer

This is the version 1.0 of XGBoost high-resolution vegetation mapping program. The codes are preliminary, so errors and faults are still possible. If you find any problem in any code, please feel free to send the authors emails. We are also appreciative of your suggestions.