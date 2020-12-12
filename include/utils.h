//
// Created by wsx on 2020/11/6.
//

#ifndef Y4_LIBTORCH_UTILS_H
#define Y4_LIBTORCH_UTILS_H

#include "ATen/core/TensorBody.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <torch/script.h>
#include <torch/optim.h>
#include <vector>



using namespace std;


class DecodeBox
{
public:
    /**************************************************************************
     * @brief 构造函数
     * @param modelPath 模型路径
     * @param t_num_anchors 每个特征层的anchor数量
     * @param t_num_classes 类别总数
     **************************************************************************/ 
    DecodeBox(float t_anchors[][2], float t_image_size[], int t_num_anchors=3, int t_num_classes=2);

    /**************************************************************************
     * @brief 预处理特征层
     * @param input 模型输出的原始特征层
     **************************************************************************/ 
    at::Tensor find_obj(at::Tensor input);

private:

    /**************************************************************************
     * @brief 用来存放每个特征层的先验框
     **************************************************************************/ 
    float anchors[3][2];

    /**************************************************************************
     * @brief 用来存储原始图像的宽高值
     **************************************************************************/ 
    float image_size[2];

    /**************************************************************************
     * @brief 代表每个特征层的先验框数量，yolov4中设为3
     **************************************************************************/ 
    int num_anchors = 3;

    /**************************************************************************
     * @brief 代表需要检测的种类的数量，按需修改
     **************************************************************************/ 
    int num_classes = 2;

    /**************************************************************************
     * @brief 代表每个bbox包含的属性的数量，bbox_attrs = num_classes + 5
     **************************************************************************/ 
    int bbox_attrs = num_classes + 5;

};

/**************************************************************************
 * @brief 调整图片大小，尺寸比例无失真
 * @param image  原始图片
 * @param size[] 目标图像的大小
 **************************************************************************/ 
cv::Mat letterbox_image(cv::Mat image, float size[]);


/**************************************************************************
 * @brief 非极大抑制NMS
 * @param output      所有特征层的输出
 * @param num_classes 类别总数量
 * @param conf_thres  置信度阈值，用于初步筛选目标框
 * @param nms_thres   nms阈值，用以挑选最终的目标框
 * @return 筛选出来的目标框 shape为(n, 6)
 **************************************************************************/ 
vector<at::Tensor> yolo_nms(at::Tensor output, int num_classes, float conf_thres=0.5, float nms_thres=0.4);


/**************************************************************************
 * @brief 去除灰条 恢复出原始的图片大小，并标定bbox的位置
 * @param box_ymin  y坐标的最小值  --tensor
 * @param box_xmin  x坐标的最小值  --tensor
 * @param box_ymax  y坐标的最大值  --tensor
 * @param box_xmax  y坐标的最大值  --tensor
 * @param model_image_size  输入模型的图片长宽
 * @param src_image_size    原始图片的长宽
 * @return 筛选出来的目标框 shape为(n, 6)
 **************************************************************************/ 
at::Tensor yolo_correct_boxes(at::Tensor box_ymin, at::Tensor box_xmin, at::Tensor box_ymax, at::Tensor box_xmax, at::Tensor model_image_size, at::Tensor src_image_size);





#endif //Y4_LIBTORCH_UTILS_H
