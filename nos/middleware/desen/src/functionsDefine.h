#ifndef FUNCTIONS_DEFINFE_H_
#define FUNCTIONS_DEFINFE_H_

#pragma once

extern "C" void YuvToRgbCuda(unsigned char* d_yuv, unsigned char* d_rgb, int width, int height);

extern "C" void RgbToYuvCuda(unsigned char* d_rgb, unsigned char* d_yuv, int width, int height);

extern "C" void Nv12ToRgbCuda(unsigned char* d_rgb, unsigned char* d_yuv, int width, int height);

extern "C" void SetYUVImageBlackCuda(unsigned char* yuv_ptr, int width, int height, int start_x, int start_y, int r_w, int r_h);

extern "C" void SetRGBImageBlackCuda(unsigned char* rgb_ptr, int width, int height, int start_x, int start_y, int r_w, int r_h);

#endif