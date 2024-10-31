#ifndef GPU_CONVERT_H
#define GPU_CONVERT_H
#include <string>
#include <fstream>
#include <memory>
#include <vector>

#include "nvscibuf.h"
#include "hw_nvmedia_gpu_common.hpp"
namespace gpuutils
{
bool save_rgbgpu_to_file(const std::string& file, RGBGPUImage* gpu, cudaStream_t stream);

/* bool save_nv12gpu_to_file(const string& file, YUVGPUImage* gpu, cudaStream_t stream); */

void free_yuv_gpu_image(YUVGPUImage* p);


void free_rgb_gpu_image(RGBGPUImage* p);

void copy_nv12_host_to_gpu(const NV12HostImage* yuv, YUVGPUImage* gpu, unsigned int ibatch, unsigned int crop_width, unsigned int crop_height, cudaStream_t stream = nullptr);

YUVGPUImage* create_yuv_gpu_image(int width, int height, int batch_size, YUVFormat format,void* luma=nullptr,void* chroma=nullptr);

RGBGPUImage* create_rgb_gpu_image(int width, int height, int batch, PixelLayout layout, netaos::gpu::DataType dtype,void* data=nullptr);

void batched_convert_yuv_to_rgb(
        YUVGPUImage* input, RGBGPUImage* output,
        int scaled_width, int scaled_height,
        int output_xoffset, int output_yoffset, FillColor fillcolor,
        float mean0,  float mean1,  float mean2,
        float scale0, float scale1, float scale2,
        Interpolation interp,
        cudaStream_t stream
        );
} // namespace gpuutils

#endif
