#include <stdio.h>
#include <fstream>
#include <vector>
#include "converter.h"

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char* file) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

std::vector<uint8_t> load_file(const std::string& file) {

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

bool save_file(const std::string& file, const void* ptr, size_t bytes) {

    std::ofstream out(file, std::ios::out | std::ios::binary);
    if (!out.is_open())
        return false;

    out.write((const char*)ptr, bytes);
    out.close();
    return out.good();
}

static __device__ uchar3 __forceinline__ rgb2yuv(unsigned char r, unsigned char g, unsigned char b) {
    unsigned char y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    unsigned char u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    unsigned char v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    return make_uchar3(y, u, v);
}

static __global__ void rgb2bl(uchar3* rgb, cudaSurfaceObject_t luma, cudaSurfaceObject_t chroma, int width, int height) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height)
        return;

    uchar3 pixel = rgb[iy * width + ix];
    uchar3 yuv = rgb2yuv(pixel.x, pixel.y, pixel.z);
    surf2Dwrite(yuv.x, luma, ix, iy);
    surf2Dwrite(make_uchar2(yuv.y, yuv.z), chroma, ix - (ix % 2), iy / 2);
}

void DoRgbToBl(uchar3* rgb, cudaSurfaceObject_t luma, cudaSurfaceObject_t chroma, int width, int height, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgb2bl<<<grid, block, 0, stream>>>(rgb, luma, chroma, width, height);
    auto status = cudaPeekAtLastError();
    if (cudaSuccess != status) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(status));
    }
}

// int main() {

//     int width = 1024;
//     int height = 1024;
//     cudaStream_t stream = nullptr;
//     checkRuntime(cudaStreamCreate(&stream));

//     cudaArray_t luma_array, chroma_array;
//     cudaChannelFormatDesc YplaneDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
//     checkRuntime(cudaMallocArray(&luma_array, &YplaneDesc, width, height, 0));

//     // One pixel of the uv channel contains 2 bytes
//     cudaChannelFormatDesc UVplaneDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
//     checkRuntime(cudaMallocArray(&chroma_array, &UVplaneDesc, width / 2, height / 2, 0));

//     cudaSurfaceObject_t luma, chroma;
//     cudaResourceDesc luma_desc = {};
//     luma_desc.resType = cudaResourceTypeArray;
//     luma_desc.res.array.array = luma_array;
//     checkRuntime(cudaCreateSurfaceObject(&luma, &luma_desc));

//     cudaResourceDesc chroma_desc = {};
//     chroma_desc.resType = cudaResourceTypeArray;
//     chroma_desc.res.array.array = chroma_array;
//     checkRuntime(cudaCreateSurfaceObject(&chroma, &chroma_desc));

//     auto rgb = load_file("fox.binary");
//     unsigned char* rgb_device = nullptr;
//     checkRuntime(cudaMalloc(&rgb_device, rgb.size()));
//     checkRuntime(cudaMemcpyAsync(rgb_device, rgb.data(), rgb.size(), cudaMemcpyHostToDevice, stream));

//     dim3 block(32, 32);
//     dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
//     rgb2bl<<<grid, block, 0, stream>>>(reinterpret_cast<uchar3*>(rgb_device), luma, chroma, width, height);
//     checkRuntime(cudaPeekAtLastError());

//     unsigned char* pl_y = nullptr;
//     unsigned char* pl_uv = nullptr;
//     checkRuntime(cudaMallocHost(&pl_y, width * height));
//     checkRuntime(cudaMallocHost(&pl_uv, width * height / 2));
//     checkRuntime(cudaMemcpy2DFromArray(pl_y, width, luma_array, 0, 0, width, height, cudaMemcpyDeviceToHost));
//     checkRuntime(cudaMemcpy2DFromArray(pl_uv, width, chroma_array, 0, 0, width, height / 2, cudaMemcpyDeviceToHost));
//     checkRuntime(cudaStreamSynchronize(stream));

//     printf("Save to pl_y.binary, pl_uv.binary\n");
//     save_file("pl_y.binary", pl_y, width * height);
//     save_file("pl_uv.binary", pl_uv, width * height / 2);

//     checkRuntime(cudaFreeHost(pl_y));
//     checkRuntime(cudaFreeHost(pl_uv));
//     checkRuntime(cudaFree(rgb_device));
//     checkRuntime(cudaDestroySurfaceObject(chroma));
//     checkRuntime(cudaDestroySurfaceObject(luma));
//     checkRuntime(cudaFreeArray(chroma_array));
//     checkRuntime(cudaFreeArray(luma_array));
//     checkRuntime(cudaStreamDestroy(stream));

//     printf("Done.\n");
//     return 0;
// }