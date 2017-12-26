# 优达学城毕业项目——Kaggle：Dog Breed Identification

## 项目来源
项目来自Kaggle的[Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)。

## 配置软硬件环境
### 购买硬件环境
1. 购买ucloud GPU主机G2系列（NVidia P40，8核，32GB内存，Windows 2012 64位 EN）按时付费。整个开发和调试过程会很漫长，而购买的计算机并没有一直在进行我们所需的计算，按时付费（按小时计费）对于个人来说，会比较划算。
2. 购买ucloud 云硬盘120GB，按月付费。因为主机是按时付费，一次使用之后会删除主机。而，整个算法所需的原始数据，处理过程的中间数据，以及一些软件安装包等东西每次都重新下载会非常耗费时间——时间就是金钱，用云硬盘用来存储这些数据是一种比较好的方式。
### 配置软件环境
1. 安装驱动
    - 安装Nvidia驱动。384.66是支持K80的版本。输入以下命令（中途遇到的确认全部选“Accept”， “Yes”， “OK”等积极的词）：
    ```
    cd ~
    sudo yum install -y gcc kernel-devel-`uname -r`
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.66/NVIDIA-Linux-x86_64-384.66.run
    sudo /bin/bash ./NVIDIA-Linux-x86_64-384.66.run
    ```
    - 查看Nvidia驱动安装是否正常。输入命令：```nvidia-smi```，应该返回GPU的信息。
    - 安装CUDA Repo，输入以下命令：
    ```
    wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-rhel6-8-0-local-ga2-8.0.61-1.x86_64-rpm
    sudo rpm -i cuda-repo-rhel6-8-0-local-ga2-8.0.61-1.x86_64-rpm
    ```
    - 安装CUDA Toolkit，输入以下命令：
    ```
    sudo yum install cuda-toolkit-8-0
    export PATH=$PATH:/usr/local/cuda-8.0/bin
    ```
    - 编辑文件```~/.bashrc```。输入命令：```vim ~/.bashrc```，打开vim。按```i```进入编辑模式。在文件最后添加一行：```export PATH=$PATH:/usr/local/cuda-8.0/bin```，文件最终内容如下：
    ```
    # .bashrc
    # Source global definitions
    if [ -f /etc/bashrc ]; then
            . /etc/bashrc
    fi

    # User specific aliases and functions
    export PATH=$PATH:/usr/local/cuda-8.0/bin
    ```
    先按```Esc```，再按```:```（冒号），进入vim编辑器的命令模式。输入```wq!```，保存并退出编辑器。
    - 执行文件```~/.bashrc```，输入命令：```source ~/.bashrc```。
    - 查看CUDA Toolkit安装是否正常。输入命令：```nvcc -–version```（直接复制粘贴到Linux控制台会有问题，建议手动输入这个命令），应该返回nvcc的版本信息。
2. 配置cuDNN v6.0 for CUDA 8.0。
    - 下载并解压。注册NVidia官网(https://developer.nvidia.com/)，然后到cuda下载页面(https://developer.nvidia.com/rdp/cudnn-download)，展开“Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0”，下载“cuDNN v6.0 Library for Linux”。在浏览器开始下载之后，复制浏览器里面下载项的地址（下载连接后面会有常常的一串token，这个token会过期，所以本文中的连接复制粘贴到Linux控制台使用），输入以下命令：
    ```
    wget http://developer2.download.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/cudnn-8.0-linux-x64-v6.0.tgz?uNqk6x1lq601bxnQ1sKzoOtcOTRMBtneEU7XIYR8oE9VoCAocU-Cvb8_l_l6HLTT4EfEdJSUSr8hBUk-iHU8irrF8oNXfQnCSTvhi2ahZpMKSHAfRWbhV-hZa2IHwHVs3jEHstpbo5--SLR1KGv8Lr4-TO9vibwTUPhmdROrtydW__57jGmtye7rMXZ_1eannPxStg9G
    mv cudnn-8.0-linux-x64-v6.0.tgz* cudnn-8.0-linux-x64-v6.0.tgz
    tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
    ```
    - 复制相应的cuDNN v6.0文件到cuda 8.0的安装目录。输入以下命令：
    ```
    sudo cp /home/ec2-user/cuda/include/cudnn.h /usr/local/cuda-8.0/include
    sudo cp /home/ec2-user/cuda/lib64/* /usr/local/cuda-8.0/lib64
    ```

3. 安装Anaconda
    - 从官网(https://www.anaconda.com/download/#linux)下载安装包，输入以下命令（因为版本一直在更新，请根据实际的下载链接下载对应的安装包）：
    ```
    wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
    ```
    - 根据官方安装指南(https://docs.anaconda.com/anaconda/install/linux)安装Anaconda。注意：第二步，因为安装包是下载到根目录的，所以安装命令为：```bash Anaconda3-5.0.1-Linux-x86_64.sh```，另外，不要在命令前面加```sudo```，加```sudo```会导致anaconda被安装到admin管理员的根目录下，而不是当前user的根目录（/home/ec2-user）。第七步，推荐选“YES”。第九步，执行命令```source ~/.bashrc```，而不是关闭控制台连接。
    - 查看Anaconda安装是否正常。输入命令：```conda --version```（直接复制粘贴到Linux控制台会有问题，建议手动输入这个命令），应该返回conda的版本信息。

4. 安装tensorflow-gpu
    - 输入以下命令，创建名为```python35```的conda环境（python3.6兼容有点问题，每次运行都会报warning）：
    ```
    conda create -n python35 python=3.5
    ```
    - 输入以下命令，激活环境```python35```：
    ```
    source activate python35
    ```
    - 检查python版本是3.5。输入以下命令：
    ```
    python --version
    ```
    - 输入以下命令，安装tensorflow-gpu包：
    ```
    conda install tensorflow-gpu
    ```
    - 验证tensorflow-gpu包安装正确和整个的硬件环境配置正确。输入以下命令：```python```，然后在这个python环境下依次运行以下python程序，确认能够正常输出“Hello World！”：**（验证这个步骤非常重要。）**
    ```
    import tensorflow as tf
    hello_constant = tf.constant('Hello World!')
    with tf.Session() as sess:
        output = sess.run(hello_constant)
        print(output)
    ```
    - 退出python35环境，输入以下命令：
    ```
    source deactivate
    ```

5. 删除多余的安装包
    - 删除除文件夹```anaconda3```外的其他的安装包文件。因为文件夹```anaconda3```是前面安装anaconda的时候创建的。输入以下命令：
    ```
    shopt -s extglob
    rm -fr !(anaconda3)
    ```

6. 安装其他的重要的包
    - 安装Linux环境下一些常用的包：
    ```
    sudo yum install git
    ```
    - 安装conda下一些常用的包：
    ```
    source activate python35
    pip install keras jupyter notebook
    ```

7. 安装和配置远程访问jupyter notebook
    参考jupyter官网(http://jupyter-notebook.readthedocs.io/en/latest/public_server.html)，简要的配置如下：
    - 生成文件```jupyter_notebook_config.py```（文件在用户主目录下的文件夹：```~/.jupyter```）。输入以下命令：
    ```
    source activate python35
    jupyter notebook --generate-config
    ```
    - 设置登陆密码，输入以下命令，并填写密码：
    ```
    jupyter notebook password
    ```
    - 启动jupyter notebook，输入以下命令：
    ```
    jupyter notebook --ip=0.0.0.0
    ```
    - 在自己本地的Windows电脑上用浏览器打开连接(http://your_server_ip:8888)（your_server_ip是实际的服务器地址），并填写前面设置的密码。

8. 制作镜像
    - 整个的软件环境的安装配置是一个步骤繁多，耗时巨大的事情，所以制作一个操作系统镜像。以后，就可以直接从镜像创建主机，几分钟就可以弄好（其实这几分钟也主要是等待主机初始化完成）。

## 开始运行
**优达学城项目审阅人员注意：程序运行成功的纪录是文件名对应的HTML文件，HTML文件名=ipython文件名+时间字符串。**
1. 启动Anaconda，进去创建的gpu环境（前面有安装tensorflow_gpu包）。
2. 预处理数据。将```1. Preprocess-GroupImages.ipynb```文件运行一遍（大概10分钟）。
3. 提取特征。将```2. Feature_extraction_from_VGG16_to_InceptionV3.ipynb```文件运行一遍（大概一个小时）。
4. 对特征分类并预测测试集的结果。将```3. Train-Predict```文件运行一遍。
5. InceptionV3 Fine-tune。将```3. InceptionV3 Fine Tune```文件运行一遍（大概一个半小时）。

## 其他说明
- 除了GitHub中的代码，项目中用到的数据请到[Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)。项目中数据，运行历史等数据并没有提交到GitHub，但是并不影响整个项目正常运行。项目中用的子文件主要有input、log、model和output。

## 参考文献
1. http://www.qyjohn.net/?p=4291
2. https://docs.anaconda.com/anaconda/install/linux
3. https://developer.nvidia.com/rdp/cudnn-download
4. http://jupyter-notebook.readthedocs.io/en/latest/public_server.html

