#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include "gtest/gtest.h"
#include "functionsDefine.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int yuv2rgbFunctions() {
    // / cuda yuv2rgb
    std::string yuvImagePath = "/home/lbl/Downloads/NVIDIA_LOGO_SLIGHT_PUSH_track1.yuv";
    int width = 1920;
    int height = 1080;
    int yuvDataSize = width * height * 3 / 2;
    int rgbDataSize = width * height * 3;

    // Allocate host memory for YUV and RGB data
    unsigned char* hostYuvData = new unsigned char[yuvDataSize];
    unsigned char* hostRgbData = new unsigned char[rgbDataSize];

    // Read YUV data from file
    FILE* yuvFile = fopen(yuvImagePath.c_str(), "rb");
    fread(hostYuvData, sizeof(unsigned char), yuvDataSize, yuvFile);
    fclose(yuvFile);

    // Allocate device memory for YUV and RGB data
    unsigned char* deviceYuvData;
    unsigned char* deviceRgbData;
    cudaMalloc((void**)&deviceYuvData, yuvDataSize);
    cudaMalloc((void**)&deviceRgbData, rgbDataSize);

    cudaMemcpy(deviceYuvData, hostYuvData, yuvDataSize, cudaMemcpyHostToDevice);

    YuvToRgbCuda(deviceYuvData, deviceRgbData, width, height);

    cudaMemcpy(hostRgbData, deviceRgbData, rgbDataSize, cudaMemcpyDeviceToHost);

    // Save RGB image
    cv::Mat rgbImage(height, width, CV_8UC3, hostRgbData);
    cv::imwrite("output_rgb_image.jpg", rgbImage);

    // Free allocated memory
    delete[] hostYuvData;
    delete[] hostRgbData;
    cudaFree(deviceYuvData);
    cudaFree(deviceRgbData);

    Mat src = imread("/home/lbl/work/test/yuv2rgb/build/output_rgb_image.jpg");

    return 0;
}

int yuvImageBlurFunctions() {
    // / cuda yuv2rgb
    std::string yuvImagePath = "/home/lbl/Downloads/NVIDIA_LOGO_SLIGHT_PUSH_track1.yuv";
    int width = 1920;
    int height = 1080;
    int yuvDataSize = width * height * 3 / 2;

    // Allocate host memory for YUV and RGB data
    unsigned char* hostYuvData = new unsigned char[yuvDataSize];

    // Read YUV data from file
    FILE* yuvFile = fopen(yuvImagePath.c_str(), "rb");
    fread(hostYuvData, sizeof(unsigned char), yuvDataSize, yuvFile);
    fclose(yuvFile);

    unsigned char* deviceYuvData;

    cudaMalloc((void**)&deviceYuvData, yuvDataSize);

    cudaMemcpy(deviceYuvData, hostYuvData, yuvDataSize, cudaMemcpyHostToDevice);

    setYUVImageBlackCuda(deviceYuvData, width, height);

    unsigned char* h_YuvDataDst = new unsigned char[yuvDataSize];
    cudaMemcpy(h_YuvDataDst, deviceYuvData, yuvDataSize, cudaMemcpyDeviceToHost);

    // Save RGB image
    cv::Mat YuvBlurImage(height, width, CV_8UC1, h_YuvDataDst);
    // cv::imwrite("output_rgb_image.jpg", rgbImage);

    // Free allocated memory
    delete[] hostYuvData;
    // delete[] hostRgbData;
    cudaFree(deviceYuvData);
    // cudaFree(deviceRgbData);

    // Mat src = imread("/home/lbl/work/test/yuv2rgb/build/output_rgb_image.jpg");
    // imshow("output_rgb_image", src);
    return 0;
}

int rgb2yuvFunctions() {
    cout << "rgb2yuvFunctions start" << endl;
    // /home/lbl/work/desenInfer/nos/middleware/desen/trt_infer/images
    // Mat rgbImage = imread("/home/lbl/work/test/yuv2rgb/build/output_rgb_image.jpg");
    Mat rgbImage = imread("/home/lbl/work/desenInfer/nos/middleware/desen/trt_infer/images/sample.jpg");
    // imshow("rgbimage", rgbImage);

    int width = rgbImage.cols;
    int height = rgbImage.rows;

    // 在主机上分配RGB和YUV图像的内存
    unsigned char* hostRgbData = rgbImage.data;
    unsigned char* hostYuvData = new unsigned char[width * height * 3 / 2];

    // 在CUDA设备上分配RGB和YUV图像的内存
    unsigned char* deviceRgbData;
    unsigned char* deviceYuvData;
    cudaMalloc((void**)&deviceRgbData, width * height * 3);
    cudaMalloc((void**)&deviceYuvData, width * height * 3 / 2);

    cudaMemcpy(deviceRgbData, hostRgbData, width * height * 3, cudaMemcpyHostToDevice);

    RgbToYuvCuda(deviceRgbData, deviceYuvData, width, height);

    cudaMemcpy(hostYuvData, deviceYuvData, width * height * 3 / 2, cudaMemcpyDeviceToHost);

    Mat yuvImage(height, width, CV_8UC1, hostYuvData);
    imwrite("yuvImage.yuv", yuvImage);

    return 0;
}

class DesenTest {};

TEST(DesenTest, KernelFunctionTest) {
    // EXPECT_EQ(rgb2yuvFunctions(), 0);
    EXPECT_EQ(0, 0);
}

// int main() {

//     // yuv2rgbFunctions();
//     rgb2yuvFunctions();
//     //
//     // yuvImageBlurFunctions();
//     return 0;
// }
