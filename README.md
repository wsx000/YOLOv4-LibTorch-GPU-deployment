# YOLOv4-LibTorch-GPU
This is a YOLOv4 post-processing code used gpu based on pure libtorch API  

环境：  
Ubuntu18.04  
CUDA10.2  
PyTorch1.7.1  
LibTorch1.7.1  
torchvision0.8.2  
OpenCV3.4.10  
RTX2070  

用法：  
1、使用pytorch2libtorch.py脚本转化模型文件，该脚本用来参考，具体代码请按照实际情况修改  
2、在下载好的文件中新建module文件夹，放入你的***.pt模型文件，新建build文件夹  
3、在cmakelists.txt 文件中修改set(CMAKE_PREFIX_PATH "/path/to/libtorch; /path/to/torchvision")  
4、在yolo.h文件中修改all_anchors、class_name、smodel_image_size参数； 修改utils.h中的num_classes参数，修改方法代码中有注释  
5、修改main.c中的modulePath为你的***.pt模型文件路径  
6、cd到2中的build文件夹中，在终端中输入以下命令：  
cmake ..  
make  
7、运行以下命令即可运行：  
./yolov4  
  
说明：  
1、main.c文件中有 单张图片检测 和 摄像头视频检测 两种方式，使用单张图片检测时需要按需修改图片路径  
2、因为代码中使用了torch官方的nms库，所以torchvision是必须的，torchvision需要自行下载源码并编译安装，放法可参考：https://blog.csdn.net/Flag_ing/article/details/109708155  
3、OpenCV的安装方法可参考：https://blog.csdn.net/Flag_ing/article/details/109508374  

运行效果测试：  
使用GPU（RTX2080Ti）下，单帧图片检测全程大概32ms  
CPU的检测全程大概1.6s  

2021.3 更新  
注意文件utils.cpp里第172行代码有问题，我比较懒，就先不改啦，具体怎么改可看Issue里的提问哈~感谢ID：happyboyneu找出的这个bug  
