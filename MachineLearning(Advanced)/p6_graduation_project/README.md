# 优达学城毕业项目——Kaggle：Dog Breed Identification

## 项目来源
项目来自Kaggle的[Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)。

## 开始运行
**优达学城项目审阅人员注意：程序运行成功的纪录是文件名对应的HTML文件，HTML文件名=ipython文件名+时间字符串。**
1. 启动Anaconda，进去创建的gpu环境（前面有安装tensorflow_gpu包）。
2. 预处理数据。将```1. Preprocess-GroupImages.ipynb```文件运行一遍（大概10分钟）。
3. 预处理数据。从Darknet官方下载model，然后，将 Darknet YOLO_v2 model转换为Keras model（参考https://www.jianshu.com/p/3e77cefeb49b）。在YOLO源代码目录YAD2K下，运行我自己写的保存狗所在区域图片的代码：```process_dog.py```。
4. 对data_train、data_val和data_test提取特征。将```2. 特征提取_从VGG16到InceptionResNetV2.ipynb```文件运行一遍（大概6个小时）。
5. 对特征分类并预测测试集的结果。将```3. Train-Predict```文件运行一遍（大概8.5个小时）。运行结果保存的HTML文件为```Dog_Breed_Identification_Train-Predict_20180222_211320_3860.html```。
6. 对yolo_data_train、yolo_data_val和yolo_data_test提取特征。将```2. 特征提取_从VGG16到InceptionResNetV2.ipynb```文件运行一遍（大概6个小时）。
7. 对特征分类并预测测试集的结果。将```3. Train-Predict```文件运行一遍（大概8.5个小时）。运行结果保存的HTML文件为```Dog_Breed_Identification_Train-Predict_20180222_141726_4092.html```。

## 其他说明
- 除了GitHub中的代码，项目中用到的数据请到[Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)。项目中数据，运行历史等数据并没有提交到GitHub，但是并不影响整个项目正常运行。项目中用的子文件主要有input、log、model和output。

## 参考文献
1. http://www.qyjohn.net/?p=4291
2. https://docs.anaconda.com/anaconda/install/linux
3. https://developer.nvidia.com/rdp/cudnn-download
4. http://jupyter-notebook.readthedocs.io/en/latest/public_server.html
5. https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07
6. https://www.jianshu.com/p/3e77cefeb49b


