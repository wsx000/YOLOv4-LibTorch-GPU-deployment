cmake_minimum_required(VERSION 3.1)
project(yolo4)

# 指定libtorch以及torchvision的路径
set(CMAKE_PREFIX_PATH "/home/curious/libtorch171/libtorch;/home/curious/libtorch171/torchvision")
# Release模式
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch not found!")
endif(NOT Torch_FOUND)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif(NOT OpenCV_FOUND)

if(NOT TorchVision_FOUND)
    message(FATAL_ERROR "TorchVision not found!")
endif(NOT TorchVision_FOUND)

# 加入头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(yolov4 src/main.cpp src/utils.cpp src/yolo.cpp)
target_link_libraries(yolov4 PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS} TorchVision::TorchVision)
set_property(TARGET yolov4 PROPERTY CXX_STANDARD 14)

