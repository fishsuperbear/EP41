#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "functionsDefine.h"
#include <stdio.h>

__device__ __host__ void YuvToRgb(unsigned char y, unsigned char u, unsigned char v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    int rTmp = (int)(y + 1.402 * (v - 128));
    int gTmp = (int)(y - 0.344136 * (u - 128) - 0.714136 * (v - 128));
    int bTmp = (int)(y + 1.772 * (u - 128));

    *r = (unsigned char)(rTmp < 0 ? 0 : (rTmp > 255 ? 255 : rTmp));
    *g = (unsigned char)(gTmp < 0 ? 0 : (gTmp > 255 ? 255 : gTmp));
    *b = (unsigned char)(bTmp < 0 ? 0 : (bTmp > 255 ? 255 : bTmp));
}

__global__ void YuvToRgbKernel(unsigned char* yuv, unsigned char* rgb, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
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

__global__ void setPixelToZero(unsigned char* img, int width, int height, int x1, int y1, int x2, int y2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        int index = y * width + x;
        img[index * 3] = 255;     // Red channel
        img[index * 3 + 1] = 255; // Green channel
        img[index * 3 + 2] = 255; // Blue channel
    }
}

__global__ void zeroYUV(unsigned char* yuvData, int width, int height, int startX, int startY, int blockWidth, int blockHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < startX + blockWidth && y >= startY && y < startY + blockHeight) {
        // 计算像素在YUV分量中的位置
        int pos = y * width + x; // Y分量位置
        int posU = (y / 2) * (width / 2) + (x / 2) + width * height; // U分量位置
        int posV = (y / 2) * (width / 2) + (x / 2) + width * height + (width / 2) * (height / 2); // V分量位置

        // 将像素值置零
        yuvData[pos] = 255; // Y分量置零
        yuvData[posU] = 128; // U分量置中值
        yuvData[posV] = 128; // V分量置中值
    }
}

__device__ __host__ void RgbToYuv(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* u, unsigned char* v)
{
    int yTmp = (int)(0.299 * r + 0.587 * g + 0.114 * b);
    int uTmp = (int)(-0.147 * r - 0.289 * g + 0.436 * b + 128);
    int vTmp = (int)(0.615 * r - 0.515 * g - 0.100 * b + 128);

    *y = (unsigned char)(yTmp < 0 ? 0 : (yTmp > 255 ? 255 : yTmp));
    *u = (unsigned char)(uTmp < 0 ? 0 : (uTmp > 255 ? 255 : uTmp));
    *v = (unsigned char)(vTmp < 0 ? 0 : (vTmp > 255 ? 255 : vTmp));
}

__global__ void RgbToYuvKernel(unsigned char* rgb, unsigned char* yuv, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
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


extern void YuvToRgbCuda(unsigned char* d_yuv, unsigned char* d_rgb, int width, int height)
{
    // unsigned char* d_yuv;
    // unsigned char* d_rgbtemp;

    // cudaMalloc((void**)&d_yuv, width * height * 3 / 2);
    // cudaMalloc((void**)&d_rgbtemp, width * height * 3);

    // cudaMemcpy(d_yuv, h_yuv, width * height * 3 / 2, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    YuvToRgbKernel<<<dimGrid, dimBlock>>>(d_yuv, d_rgb, width, height);

    // setImageBlackCuda(d_rgbtemp, d_rgb, width, height);
    // setPixelToZero<<<gridSize, blockSize>>>(d_img, image.cols, image.rows, image.cols/2-200, image.rows/2 -200, image.cols/2 + 200, image.rows/2 + 200);
    setPixelToZero<<<dimGrid, dimBlock>>>(d_rgb, width, height, width/2 -200, height/2 -200, width/2 +200, height /2 + 200);
    // cudaMemcpy(h_rgb, d_rgb, width * height * 3, cudaMemcpyDeviceToHost);

    // cudaFree(d_rgbtemp);
    // cudaFree(d_rgb);
}


extern void RgbToYuvCuda(unsigned char* d_rgb, unsigned char* d_yuv, int width, int height)
{
    
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
__global__ void zeroPixels(float* yuvData, int width, int height, int startX, int startY, int regionWidth, int regionHeight) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查像素位置是否在指定的区域内
    if (row >= startY && row < startY + regionHeight && col >= startX && col < startX + regionWidth) {
        int index = row * width + col;
        
        // 将YUV像素值置零
        yuvData[index] = 0;
        yuvData[index + width * height] = 0.5;
        int yuvindex = index + width * height * 1.25;
        yuvData[yuvindex] = 0.5;
    }
}

extern void setYUVImageBlackCuda(unsigned char* d_src_rgb,  int width, int height)
{
    printf("setYUVImageBlackCuda  start \n");
    // std::cout <<"RgbToYuvCuda function start" <<std::endl; 
    // unsigned char* d_rgb;
    // unsigned char* d_yuv;

    // cudaMalloc((void**)&d_rgb, width * height * 3);
    // cudaMalloc((void**)&d_yuv, width * height * 3 / 2);

    // cudaMemcpy(d_rgb, h_rgb, width * height * 3, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // 调用CUDA内核函数
    // zeroPixels<<<gridSize, blockSize>>>((float *)d_src_rgb, width, height, 100, 100, 200, 200);
    zeroYUV<<<gridSize, blockSize>>> (d_src_rgb, width, height, 100, 100, 200, 200);
    // setPixelToZero<<<dimGrid, dimBlock>>>(d_src_rgb, d_dst_rgb, width, height);

    // cudaMemcpy(h_yuv, d_yuv, width * height * 3 / 2, cudaMemcpyDeviceToHost);
    printf("setYUVImageBlackCuda  end \n");
    // cudaFree(d_rgb);
    // cudaFree(d_yuv);

}