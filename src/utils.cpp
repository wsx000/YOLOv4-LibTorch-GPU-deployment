//
// Created by wsx on 2020/11/6.
//

#include "utils.h"
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/ScalarType.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "torchvision/cpu/vision_cpu.h"
// #include "torchvision/cuda/vision_cuda.h"
#include "yolo.h"

using namespace at::indexing;
using namespace std;

/**************************************************************************
 * @brief 构造函数
 * @param modelPath 模型路径
 **************************************************************************/ 
DecodeBox::DecodeBox(float t_layer_anchors[][2], float t_model_image_size[])
{
    // 获取当前特征图的anchor参数
    for(size_t i=0; i<3; i++)
        for (size_t j = 0; j < 2; j++)
            anchors[i][j] = t_layer_anchors[i][j];

    // 获取模型中图像的尺寸
    for (size_t i = 0; i < 2; i++)
        image_size[i] = t_model_image_size[i];
}


/**************************************************************************
 * @brief 预处理特征层
 * @param input 模型输出的原始特征层
 **************************************************************************/ 
at::Tensor DecodeBox::find_obj(at::Tensor input) 
{
    // 获取尺寸等参数
    int batch_size = input.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);
    // 步长
    int stride_w = image_size[0] / input_width;
    int stride_h = image_size[1] / input_height;

    // 将anchor映射到栅格图中
    float scaled_anchors[3][2];
    for (int i=0; i<3; i++)
    {
        scaled_anchors[i][0] = anchors[i][0] / stride_w;
        scaled_anchors[i][1] = anchors[i][1] / stride_h;
    }

    // (bs, 3*(5+num_classes), h, w)  -->  (bs, 3, h, w, (5+num_classes))
    at::Tensor prediction = input.view({batch_size, num_anchors, bbox_attrs, input_height, input_width}).permute({0, 1, 3, 4, 2}).contiguous();

    // 先验框中心位置的调整参数
    at::Tensor x = at::sigmoid(prediction.index({"...", 0}));
    at::Tensor y = at::sigmoid(prediction.index({"...", 1}));
    // 先验框宽高参数调整
    at::Tensor w = prediction.index({"...", 2});
    at::Tensor h = prediction.index({"...", 3});
    // 置信度获取
    at::Tensor conf = at::sigmoid(prediction.index({"...", 4}));
    // 物体类别置信度
    at::Tensor pred_cls = at::sigmoid(prediction.index({"...", Slice{5, None}}));
    // 生成网格  bs, 3, h, w
    at::Tensor grid_x = at::linspace(0, input_width-1, input_width).repeat({input_width, 1}).repeat({batch_size*num_anchors, 1, 1}).view({x.sizes()}).toType(torch::kFloat);
    at::Tensor grid_y = at::linspace(0, input_height-1, input_height).repeat({input_height, 1}).t().repeat({batch_size*num_anchors, 1, 1}).view({y.sizes()}).toType(torch::kFloat);

    // 生成先验框的宽高  数组转换为tensor，按列选取   最终shape  bs, 3, h, w
    at::Tensor anchor_w = at::from_blob(scaled_anchors, {3, 2}, at::kFloat).index_select(1, at::tensor(0).toType(at::kLong))\
                                        .repeat({batch_size, input_height*input_width}).view(w.sizes());
    at::Tensor anchor_h = at::from_blob(scaled_anchors, {3, 2}, at::kFloat).index_select(1, at::tensor(1).toType(at::kLong))\
                                        .repeat({batch_size, input_height*input_width}).view(h.sizes());

    // 计算调整后的先验框中心与宽高  创建全零tensor  bs, 3, h, w, 4
    at::Tensor pred_boxes = at::zeros({prediction.index({"...", Slice({None, 4})}).sizes()}).toType(at::kFloat);
    // 填充调整到栅格图上的尺寸值
    pred_boxes.index_put_({"...", 0}, (x.data() + grid_x));
    pred_boxes.index_put_({"...", 1}, (y.data() + grid_y));
    pred_boxes.index_put_({"...", 2}, (at::exp(w.data()) * anchor_w));
    pred_boxes.index_put_({"...", 3}, (at::exp(h.data()) * anchor_h));

    // 生成转换tensor  (batch_size, 6) -->  (batch_size, (x, y, w, h, conf, pred_cls))
    at::Tensor grid2org = at::tensor({stride_w, stride_h, stride_w, stride_h}).toType(at::kFloat);
    at::Tensor output = at::cat({pred_boxes.view({batch_size, -1, 4}) * grid2org, \
                                conf.view({batch_size, -1, 1}), \
                                pred_cls.view({batch_size, -1, num_classes})}, \
                                -1); 
    return output.data();
}


// 调整图片大小
cv::Mat letterbox_image(cv::Mat image, float size[])
{
    // 图片真实大小
    float iw = image.cols, ih = image.rows;
    // 网络输入图片的大小
    float w = size[0], h = size[1];
    float scale = min(w/iw, h/ih);
    // 调整后的大小
    int nw = int(iw * scale), nh = int(ih * scale);
    // 
    cv::resize(image, image, {nw, nh});
    // 创建图片
    cv::Mat new_image(w, h, CV_8UC3, cv::Scalar(128, 128, 128));
    // 设置画布绘制区域并复制
    cv::Rect roi_rect = cv::Rect((w-nw)/2, (h-nh)/2, nw, nh);
    image.copyTo(new_image(roi_rect));

    // cv::imshow("aa", new_image);
    // cv::waitKey();
    return new_image;
    
}


/**************************************************************************
 * @brief 非极大抑制NMS
 * @param output      所有特征层的输出
 * @param num_classes 类别总数量
 * @param conf_thres  置信度阈值，用于初步筛选目标框
 * @param nms_thres   nms阈值，用以挑选最终的目标框
 * @return 筛选出来的目标框 shape为(n, 6)
 **************************************************************************/ 
vector<at::Tensor> yolo_nms(at::Tensor prediction, int num_classes, float conf_thres, float nms_thres)
{
    // 创建一个与predict的shape相同的tensor，且无内容  shape： (bs, 3*(13*13+26*26+52*52)， 5+num_classes)
    at::Tensor box_corner = at::zeros(prediction.sizes());

    // 求左上角和右下角
    box_corner.index_put_({"...", 0}, prediction.index({"...", 0}) - prediction.index({"...", 2})/2);
    box_corner.index_put_({"...", 1}, prediction.index({"...", 1}) - prediction.index({"...", 3})/2);
    box_corner.index_put_({"...", 2}, prediction.index({"...", 0}) + prediction.index({"...", 2})/2);
    box_corner.index_put_({"...", 3}, prediction.index({"...", 1}) + prediction.index({"...", 3})/2);
    // 赋值 x1 y1 x2 y2
    prediction.index_put_({"...", Slice(None,4)}, box_corner.index({"...", Slice(None,4)}));

    // 存放nms后的输出
    vector<at::Tensor> nms_output;

    // bs 为1，直接提取 (3*(13*13+26*26+52*52)， 5+num_classes)
    at::Tensor output = prediction[0];

    // 提取最优目标及其置信度
    std::tuple<at::Tensor, at::Tensor> temp = at::max(output.index({"...", Slice(5, 5 + num_classes)}), 1, true);
    at::Tensor class_conf = get<0>(temp);
    at::Tensor class_pred = get<1>(temp);

    // 利用置信度进行第一轮筛选
    at::Tensor conf_mask = (output.index({"...", 4}) * class_conf.index({"...", 0}) >= conf_thres).squeeze();

    // 留下有目标的部分
    output = output.index({conf_mask});
    
    // 没检测出可能的目标，直接返回空结果
    if(output.size(0) == 0)
    {
        return nms_output;
    }

    class_conf = class_conf.index({conf_mask});
    class_pred = class_pred.index({conf_mask});

    // 获得的内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    at::Tensor detections = at::cat({output.index({"...", Slice(None, 5)}), class_conf.toType(at::kFloat), class_pred.toType(at::kFloat)}, -1);
    // 获取种类的个数，返回无重复的元素
    std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_labels_tuple = at::unique_consecutive(detections.index({"...", -1}));
    at::Tensor unique_labels = get<0>(unique_labels_tuple);


    // 遍历所有的种类
    for(int i=0; i<unique_labels.size(0); i++)
    {
        // 获取某个类初步筛选后的预测结果
        at::Tensor detections_class = detections.index({detections.index({"...", -1}) == unique_labels[i]});

        // 使用官方的NMS  类别，得分，iou阈值
        at::Tensor keep = nms_cpu(detections_class.index({"...", Slice(None,4)}), detections_class.index({"...", 4})*detections_class.index({"...", 5}), nms_thres);
        
        // 提取留下来的目标框
        at::Tensor max_detection = detections_class.index({keep});
        if(i==0)
        {
            nms_output.push_back(max_detection);
        }
        else
        {
            nms_output[0] = at::cat({nms_output[0], max_detection});
        }
    }
    return nms_output;
}



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
at::Tensor yolo_correct_boxes(at::Tensor top, at::Tensor left, at::Tensor bottom, at::Tensor right, at::Tensor model_image_size, at::Tensor src_image_size)
{

    // 计算初步缩放后的尺寸，不包含灰度条  shape (1, 2) (h ,w)
    at::Tensor new_shape = (src_image_size * at::amin(model_image_size/src_image_size, 0));
    // 计算长宽对应的缩放比例
    at::Tensor scale = model_image_size/new_shape;

    // 计算在灰度条上的偏置
    at::Tensor offset = (model_image_size-new_shape)/2./model_image_size;    // 归一化了

    // 形成中心点（y，x）坐标 shape = n,2
    at::Tensor box_yx = at::cat({(top+bottom)/2, (left+right)/2}, -1)/model_image_size;

    // 形成宽高参数 shape = n,2
    at::Tensor box_hw = at::cat({bottom-top, right-left}, -1)/model_image_size;

    // 计算恢复出来的真实图片里的中心点坐标及长宽
    box_yx = (box_yx - offset) * scale;
    box_hw = box_hw * scale;

    // 
    at::Tensor box_mins = (box_yx - box_hw/2.) * src_image_size;
    at::Tensor box_maxs = (box_yx + box_hw/2.) * src_image_size;

    // ymin ximn ymax xmax
    at::Tensor boxes = at::cat({box_mins, box_maxs}, -1);

    return boxes;
}




