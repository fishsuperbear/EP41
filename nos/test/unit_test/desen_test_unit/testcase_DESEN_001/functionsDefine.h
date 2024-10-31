#ifndef TESTCASE_DESEN_001_FUNCTIONS_DEFINFE_H_
#define TESTCASE_DESEN_001_FUNCTIONS_DEFINFE_H_

#pragma once

extern "C" void YuvToRgbCuda(unsigned char* h_yuv, unsigned char* h_rgb, int width, int height);

extern "C" void RgbToYuvCuda(unsigned char* h_rgb, unsigned char* h_yuv, int width, int height);

extern "C" void setYUVImageBlackCuda(unsigned char* d_src_rgb, int width, int height);

#endif