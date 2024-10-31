#ifndef KERNEL_PROCESS_H__
#define KERNEL_PROCESS_H__
#include "hw_nvmedia_gpu_common.hpp"

// If yuv_format == NV12BlockLinear, luma must be of type cudaTexture_t, otherwise luma must be ydata of type unsigned char*.
// If yuv_format == NV12BlockLinear, chroma must be of type cudaTexture_t, otherwise chroma must be uvdata of type unsigned char*.
// if out_layout  == NHWC_RGB or NHWC_BGR, out_stride are used, otherwise ignore out_stride
void batched_convert_yuv_to_rgb(
        const void* luma, const void* chroma, int input_width, int input_stride, int input_height, int input_batch, YUVFormat yuv_format,
        int scaled_width, int scaled_height, int output_xoffset, int output_yoffset, FillColor fillcolor,
        void* out_ptr, int out_width, int out_stride, int out_height,
        netaos::gpu::DataType out_dtype, PixelLayout out_layout, Interpolation interp,
        float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
        void* stream
        );

#endif //KERNEL_PROCESS_H__
