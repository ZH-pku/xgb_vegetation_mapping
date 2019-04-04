#Program: High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features

Please CITE the following article when using the codes: 

H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing. (submitted)

Contacts

Heng Zhang: heng.zhang@pku.edu.cn; hengzhang.zhh@gmail.com
Anwar Eziz: anwareziz@pku.edu.cn

Summary and Instructions

This directory contains codes of Landsat feature computing. The features to compute includes spectral variables, GLCM textures, color spaces, and moving window statistical values. Before computing, please change the 'root' directory in each 'main_xxxx.m' file to the current folder. Also, you should ckeck the geo-information of the input Landsat images (e.g. using ArcGIS software), and change the relevant variables in 'main_xxxx.m' accordingly. If your MATLAB has Mapping Toolbox, you may change the way of reading and writing raster images instead.

Attention: the texture and moving window statistical value calculating requires GPU computing, please make sure that you have at least one NVIDIA GPU device, and CUDA has been correctly installed in your local environment.

Disclaimer

This is the version 1.0 of XGBoost high-resolution vegetation mapping program. The codes are preliminary, so errors and faults are still possible. If you find any problem in any code, please feel free to send the authors emails. We are also appreciative of your suggestions.