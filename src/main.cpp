#include <iostream>
#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include "string"
#include "vector"
#include "ATen/Parallel.h"
#include <iomanip>
#include "chrono"
#include "yolo.h"

using namespace std;
using namespace std::chrono;

int main()
{
    // 模型路径
    const char* modulePath = "/home/curious/code/yolo4-cpp/module/111.pt";
    // 加载模型
    YOLO yolo(modulePath);

    /*=====================================单张图片========================================*/
    // const char* imgPath = "/home/curious/code/yolo4-cpp/image/62.jpg";
    // while(1)
    // {
    //     auto start = high_resolution_clock::now();
    //     // 读取图片
    //     cv::Mat image = cv::imread(imgPath);
    //     // 前向传播并推理，输出结果为：n * (ymin, xmin, ymax, xmax, conf, class) 的数组
    //     vector<vector<float>> out_boxes = yolo.detect_image(image);
    //     // 记录结束时间
    //     auto end = high_resolution_clock::now();
    //     auto duration = duration_cast<milliseconds>(end - start);
    //     cout << "inference takes：" << duration.count() << "ms" << endl;
    //     yolo.draw_rectangle(out_boxes, image, true);
    // }

    /*===================================================================================*/

    /*=====================================视频检测========================================*/
    // 打开摄像头
    cv::VideoCapture cap(0);
    while (1)
    {
        // 读取视频帧
        cv::Mat frame;
        // 读取摄像头
        cap >> frame;
        // 记录开始时间
        auto start = high_resolution_clock::now();
        // 前向传播并推理，输出结果为：n * (ymin, xmin, ymax, xmax, conf, class) 的数组
        vector<vector<float>> out_boxes = yolo.detect_image(frame);
        // 记录结束时间
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        cout << "all takes：" << duration.count() << "ms" << endl;
        // 结果显示
        yolo.show_results(out_boxes, frame, true);
    }
    cap.release();
    cv::destroyAllWindows();
    /*===================================================================================*/


    return 0;
}
