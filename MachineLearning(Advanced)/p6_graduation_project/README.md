# 优达学城毕业项目——Kaggle：Dog Breed Identification

## 配置软硬件环境
1. 购买硬件环境
- 购买ucloud GPU主机G2系列（NVidia P40，8核，16GB内存，Windows 2012 64位 EN）按时付费。整个开发和调试过程会很漫长，而购买的计算机并没有一直在进行我们所需的计算，按时付费（按小时计费）对于个人来说，会比较划算。
- 购买ucloud 云硬盘120GB，按月付费。因为主机是按时付费，一次使用之后会删除主机。而，整个算法所需的原始数据，处理过程的中间数据，以及一些软件安装包等东西每次都重新下载会非常耗费时间——时间就是金钱，用云硬盘用来存储这些数据是一种比较好的方式。
2. 配置软件环境
- 安装驱动 
**注意：本次配置环境所用的安装包是一个月之前下载的，所以，提供的安装包的下载链接地址可能会有问题。另外，不同的显卡型号配置，驱动安装包和cnn包会不一样，需要到NVidia官网搜索下载对应的东西才行。配置环境的过程问题会比较多，遇到问题可以多上网搜索一下。**
    - 安装```vc_redist.x64.exe```。安装包来自于[微软官网](https://www.microsoft.com/en-us/download/details.aspx?id=48145)，。
    - 安装显卡驱动```win2012_cuda_8.0.61_windows.exe```（1.19GB）。安装包来自于NVidia官网。
    - 安装显卡驱动补丁```win2012_cuda_8.0.61.2_windows.exe```（42MB）。安装包来自于NVidia官网。
    - 解压```cudnn-8.0-windows10-x64-v6.0_work_at_win2012.zip```。安装包来自于NVidia官网。然后将解压之后的```cudnn-8.0-windows10-x64-v6.0_work_at_win2012```文件夹下面的，```cuda```文件夹复制到```C```盘根目录。
    - 添加环境变量。```System variables```框内，```Path```变量，添加```C:\cuda\bin;```（注意环境变量关于分号的格式要求。）。
    - 验证驱动程序安装正确。Win+R，打开Windows运行窗口，然后输入```devmgmt.msc```打开设备管理器，点开Display adapter，会看到NVidia Tesla P40。
- 安装Anaconda。
    - 安装Anaconda3-4.4.0-Windows-x86_64.exe（Python 3.6，64-bit）。安装包来自于[Anaconda官网](https://www.anaconda.com/download/)。
    - 打开Anaconda终端。
    - 在Anaconda终端运行以下命令，创建名为```gpu```的conda环境：
    ```
    conda create -n gpu python=3.5
    ```
    - 在Anaconda终端运行以下命令，激活环境```gpu```：
    ```
    activate gpu
    ```
    - 在Anaconda终端运行以下命令，安装tensorflow-gpu包：
    ```
    conda install tensorflow-gpu
    ```
    - 验证tensorflow-gpu包安装正确和整个的硬件环境配置正确。在Anaconda终端运行以下命令：```python```，然后在这个python环境下依次运行以下python程序，确认能够正常输出“Hello World！”：**（验证这个步骤非常重要。）**
    ```
    import tensorflow as tf
    hello_constant = tf.constant('Hello World!')
    with tf.Session() as sess:
        output = sess.run(hello_constant)
        print(output)
    ```
    - 在Anaconda终端运行以下命令，安装其他会用到的包：
    ```
    conda install pandas matplotlib jupyter notebook scipy scikit-learn seaborn h5py
    pip install pickleshare Pillow keras tqdm opencv-python
    ```
- 安装常用软件：Firefox，git客户端，VS Code。
    - 官网下载相应的安装包，默认安装即可。
3. 制作镜像
- 整个的软件环境的安装配置是一个步骤繁多，非常耗时的事情，比如下载exe的安装包和下载conda的安装包，整个过程会消耗几个小时，所以制作一个操作系统镜像。以后，就可以直接从镜像创建主机，几分钟就可以弄好（其实这几分钟也主要是等待主机初始化完成）。

