# AWS p2.xlarge(K80) Ubuntu16.4环境配置

## References：
1. http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

GCC需要安装5.3，到gcc的官网[https://ftp.gnu.org/gnu/gcc/]下载gcc5.3再安装
1. 下载安装包：
    ```
    sudo wget https://ftp.gnu.org/gnu/gcc/gcc-5.3.0/gcc-5.3.0.tar.gz
    ```
2. 解压：
    ```
    tar -zxvf gcc-5.3.0.tar.gz
    ```
3. 下载编译所需依赖项：
    ```
    cd gcc-5.3.0  //进入解包后的gcc文件夹
    ./contrib/download_prerequisites  //下载依赖项
    cd ..  //返回上层目录
    ```
4.建立编译输出目录：
    ```
    mkdir gcc-build-5.3.0
    ```
5.进入输出目录，执行以下命令，并生成makefile文件：
    ```
    cd gcc-build-5.3.0
    ../gcc-5.3.0/configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
    ```

下载cuda：
```
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64
```

【问题】ubuntu16 configure: error: no acceptable C compiler found in $PATH

【解决】Run```sudo apt-get install build-essential```to install the C compiler.
https://askubuntu.com/questions/237576/no-acceptable-c-compiler-found-in-path


