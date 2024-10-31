#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "functionsDefine.h"

__device__ __host__ void YuvToRgb(unsigned char y, unsigned char u, unsigned char v, unsigned char* r, unsigned char* g,
                                  unsigned char* b) {
    int rTmp = (int)(y + 1.402 * (v - 128));
    int gTmp = (int)(y - 0.344136 * (u - 128) - 0.714136 * (v - 128));
    int bTmp = (int)(y + 1.772 * (u - 128));

    *r = (unsigned char)(rTmp < 0 ? 0 : (rTmp > 255 ? 255 : rTmp));
    *g = (unsigned char)(gTmp < 0 ? 0 : (gTmp > 255 ? 255 : gTmp));
    *b = (unsigned char)(bTmp < 0 ? 0 : (bTmp > 255 ? 255 : bTmp));
}

__global__ void YuvToRgbKernel(unsigned char* yuv, unsigned char* rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int yuvIndex = y * width + x;
        int rgbIndex = yuvIndex * 3;

        unsigned char yVal = yuv[yuvIndex];
        unsigned char uVal = yuv[width * height + y / 2 * (width / 2) + x / 2];
        unsigned char vVal = yuv[width * height + width * height / 4 + y / 2 * (width / 2) + x / 2];

        unsigned char r, g, b;
        YuvToRgb(yVal, uVal, vVal, &r, &g, &b);

        rgb[rgbIndex] = r;
        rgb[rgbIndex + 1] = g;
        rgb[rgbIndex + 2] = b;
    }
}

__device__ __host__ void RgbToYuv(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* u,
                                  unsigned char* v) {
    int yTmp = (int)(0.299 * r + 0.587 * g + 0.114 * b);
    int uTmp = (int)(-0.147 * r - 0.289 * g + 0.436 * b + 128);
    int vTmp = (int)(0.615 * r - 0.515 * g - 0.100 * b + 128);

    *y = (unsigned char)(yTmp < 0 ? 0 : (yTmp > 255 ? 255 : yTmp));
    *u = (unsigned char)(uTmp < 0 ? 0 : (uTmp > 255 ? 255 : uTmp));
    *v = (unsigned char)(vTmp < 0 ? 0 : (vTmp > 255 ? 255 : vTmp));
}

__global__ void RgbToYuvKernel(unsigned char* rgb, unsigned char* yuv, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rgbIndex = y * width * 3 + x * 3;
        int yuvIndex = rgbIndex / 3;

        unsigned char rVal = rgb[rgbIndex];
        unsigned char gVal = rgb[rgbIndex + 1];
        unsigned char bVal = rgb[rgbIndex + 2];

        unsigned char y, u, v;
        RgbToYuv(rVal, gVal, bVal, &y, &u, &v);

        yuv[yuvIndex] = y;
        yuv[width * height + y / 2 * (width / 2) + x / 2] = u;
        yuv[width * height + width * height / 4 + y / 2 * (width / 2) + x / 2] = v;
    }
}

__global__ void nv12ToRgbKernel(const unsigned char* nv12, unsigned char* rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int yIndex = y * width + x;
        int uvIndex = (y / 2) * width + x - (x % 2);

        unsigned char yValue = nv12[yIndex];
        unsigned char uValue = nv12[width * height + uvIndex];
        unsigned char vValue = nv12[width * height + uvIndex + 1];

        int r = yValue + 1.402 * (vValue - 128);
        int g = yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128);
        int b = yValue + 1.772 * (uValue - 128);

        r = min(max(r, 0), 255);
        g = min(max(g, 0), 255);
        b = min(max(b, 0), 255);

        int rgbIndex = yIndex * 3;
        rgb[rgbIndex] = r;
        rgb[rgbIndex + 1] = g;
        rgb[rgbIndex + 2] = b;
    }
}

void Nv12ToRgbCuda(unsigned char* d_yuv, unsigned char* d_rgb, int width, int height) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    nv12ToRgbKernel<<<dimGrid, dimBlock>>>(d_yuv, d_rgb, width, height);
}

void YuvToRgbCuda(unsigned char* d_yuv, unsigned char* d_rgb, int width, int height) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    RgbToYuvKernel<<<dimGrid, dimBlock>>>(d_yuv, d_rgb, width, height);
}

void RgbToYuvCuda(unsigned char* d_rgb, unsigned char* d_yuv, int width, int height) {

    // unsigned char* d_rgb;
    // unsigned char* d_yuv;

    // cudaMalloc((void**)&d_rgb, width * height * 3);
    // cudaMalloc((void**)&d_yuv, width * height * 3 / 2);

    // cudaMemcpy(d_rgb, h_rgb, width * height * 3, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    RgbToYuvKernel<<<dimGrid, dimBlock>>>(d_rgb, d_yuv, width, height);

    // cudaMemcpy(h_yuv, d_yuv, width * height * 3 / 2, cudaMemcpyDeviceToHost);

    // cudaFree(d_rgb);
    // cudaFree(d_yuv);
}

__global__ void whiteOutRegion(unsigned char* yData, unsigned char* uvData, int width, int height, int startX,
                               int startY, int regionWidth, int regionHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < startX + regionWidth && y >= startY && y < startY + regionHeight) {
        // 涂白Y分量
        yData[y * width + x] = 255;

        // 涂白UV分量
        if (y % 2 == 0 && x % 2 == 0) {
            int uvIndex = (y / 2) * (width / 2) + (x / 2);
            uvData[uvIndex] = 128;
            uvData[uvIndex + 1] = 128;
        }
    }
}

__global__ void zeroNV12(unsigned char* yData, unsigned char* uvData, int width, int startX, int startY, int regionW,
                         int regionH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < startX + regionW && y >= startY && y < startY + regionH) {
        // 涂白Y分量
        yData[y * width + x] = 255;
        // 涂白UV分量
        if (y % 2 == 0 && x % 2 == 0) {
            int uvIndex = (y / 2) * (width / 2) + (x / 2);
            uvData[uvIndex] = 128;
            uvData[uvIndex + 1] = 128;
        }

        // // 计算像素在YUV分量中的位置
        // int pos = y * width + x;                                                                   // Y分量位置
        // int posU = (y / 2) * (width / 2) + (x / 2) + width * height;                               // U分量位置
        // int posV = (y / 2) * (width / 2) + (x / 2) + width * height + (width / 2) * (height / 2);  // V分量位置

        // // 将像素值置零
        // yuvData[pos] = 255;   // Y分量置零
        // yuvData[posU] = 128;  // U分量置中值
        // yuvData[posV] = 128;  // V分量置中值
    }
}

void SetYUVImageBlackCuda(unsigned char* yuv_ptr, int width, int height, int start_x, int start_y, int r_w, int r_h) {
    unsigned char* y_ptr = yuv_ptr;
    unsigned char* uv_ptr = yuv_ptr + width * height;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    zeroNV12<<<gridSize, blockSize>>>(y_ptr, uv_ptr, width, start_x, start_y, r_w, r_h);
}

__global__ void setPixelKernel(unsigned char* image, int width, int x, int y, int rectWidth, int rectHeight) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= y && row < (y + rectHeight) && col >= x && col < (x + rectWidth)) {
        int index = (row * width + col) * 3;
        image[index] = 255;      // blue
        image[index + 1] = 255;  // green
        image[index + 2] = 255;  // red
    }
}

__global__ void setPixelKernel(unsigned char* image, int width, int x, int y, int rectWidth, int rectHeight,
                               int channels, unsigned char r, unsigned char g, unsigned char b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= y && row < (y + rectHeight) && col >= x && col < (x + rectWidth)) {
        int index = (row * width + col) * channels;
        image[index] = b;      // blue
        image[index + 1] = g;  // green
        image[index + 2] = r;  // red
    }
}

__global__ void setRGBPixelArea(unsigned char* image, int width, int height, int startX, int startY, int endX,
                                int endY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < endX && y >= startY && y < endY) {
        int index = y * width + x;
        image[index] = 128;
    }
}

__global__ void zeroRegionInRGBImage(unsigned char* image, int width, int height, int startX, int startY,
                                     int regionWidth, int regionHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < startX + regionWidth && y >= startY && y < startY + regionHeight) {
        int index = 3 * (y * width + x);
        image[index] = 255;      // Red
        image[index + 1] = 255;  // Green
        image[index + 2] = 128;  // Blue
    }
}

void SetRGBImageBlackCuda(unsigned char* rgb_ptr, int width, int height, int start_x, int start_y, int r_w, int r_h) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // start SetRGBImageBlackCuda  width is :3840 height is :2160 , startx is :3651 , startY is :622, imageWidth is 49 , imageHeight is :28
    zeroRegionInRGBImage<<<gridDim, blockDim>>>(rgb_ptr, width, height, start_x, start_y, r_w, r_h);

    // zeroRegionInRGBImage<<<gridDim, blockDim>>>(d_src_rgb, width, height, 200, 300, 490, 280);
}