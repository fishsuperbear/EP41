#ifndef XXX_MY_CUDA_KERNEL_H
#define XXX_MY_CUDA_KERNEL_H
#include <cuda_runtime.h>

extern "C" void DoRgbToBl(uchar3* rgb, cudaSurfaceObject_t luma, cudaSurfaceObject_t chroma, int width, int height, cudaStream_t stream);
extern "C" std::vector<uint8_t> load_file(const std::string& file);
extern "C" bool save_file(const std::string& file, const void* ptr, size_t bytes);

#endif