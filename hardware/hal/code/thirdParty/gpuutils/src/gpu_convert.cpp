#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gpu_convert.hpp"
#include <kernel_process.hpp>


#define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)

template<typename _T>struct AsDataType{};
template<>struct AsDataType<uint8_t>{static const netaos::gpu::DataType type = netaos::gpu::DataType::Uint8;};
template<>struct AsDataType<float>  {static const netaos::gpu::DataType type = netaos::gpu::DataType::Float32;};
template<>struct AsDataType<__half> {static const netaos::gpu::DataType type = netaos::gpu::DataType::Float16;};

template<netaos::gpu::DataType _T>struct AsPODType{};
template<>struct AsPODType<netaos::gpu::DataType::Uint8>   {typedef uint8_t type;};
template<>struct AsPODType<netaos::gpu::DataType::Float32> {typedef float   type;};
template<>struct AsPODType<netaos::gpu::DataType::Float16> {typedef __half  type;};


static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

size_t dtype_sizeof(netaos::gpu::DataType dtype){
    switch(dtype){
        case netaos::gpu::DataType::Float32: return sizeof(AsPODType<netaos::gpu::DataType::Float32>::type);
        case netaos::gpu::DataType::Float16: return sizeof(AsPODType<netaos::gpu::DataType::Float16>::type);
        case netaos::gpu::DataType::Uint8:   return sizeof(AsPODType<netaos::gpu::DataType::Uint8>::type);
    default: return 0;
    }
}

bool gpuutils::save_rgbgpu_to_file(const std::string& filename, RGBGPUImage* gpu, cudaStream_t stream){

    unsigned int header[] = {0xAABBCCEF, static_cast<unsigned int>(gpu->width), static_cast<unsigned int>(gpu->height), 
                                static_cast<unsigned int>(gpu->channel), static_cast<unsigned int>(gpu->batch), 
                                static_cast<unsigned int>(gpu->layout), static_cast<unsigned int>(gpu->dtype)};
    std::fstream fout(filename, std::ios::binary | std::ios::out);
    if(!fout.good()){
        std::fprintf(stderr, "Can not open %s\n", filename.c_str());
        return false;
    }
    fout.write((char*)header, sizeof(header));

    size_t num_element    = gpu->width * gpu->height * gpu->channel * gpu->batch;
    size_t sizeof_element = dtype_sizeof(gpu->dtype);
    uint8_t* phost   = new uint8_t[num_element * sizeof_element];
    checkRuntime(cudaMemcpyAsync(phost, gpu->data, num_element * sizeof_element, ::cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    fout.write((char*)phost, num_element * sizeof_element);

    delete [] phost;
    return fout.good();
}
void gpuutils::free_rgb_gpu_image(RGBGPUImage* p){
    if(p){
        if(p->data) checkRuntime(cudaFree(p->data));
        delete p;
    }
}

void gpuutils::free_yuv_gpu_image(YUVGPUImage* p){
    if(p){
        if(p->format == YUVFormat::NV12PitchLinear){
            if(p->chroma) checkRuntime(cudaFree(p->chroma));
            if(p->luma)   checkRuntime(cudaFree(p->luma));
        }else if(p->format == YUVFormat::NV12BlockLinear){
            if(p->chroma) checkRuntime(cudaDestroyTextureObject((cudaTextureObject_t)p->chroma));
            if(p->luma)   checkRuntime(cudaDestroyTextureObject((cudaTextureObject_t)p->luma));
            if(p->chroma_array) checkRuntime(cudaFreeArray((cudaArray_t)p->chroma_array));
            if(p->luma_array)   checkRuntime(cudaFreeArray((cudaArray_t)p->luma_array));
        }else if(p->format == YUVFormat::YUV422Packed_YUYV_PitchLinear){
            if(p->luma) checkRuntime(cudaFree(p->luma));
        }
        delete p;
    }
}

void gpuutils::copy_nv12_host_to_gpu(const NV12HostImage* host, YUVGPUImage* gpu, unsigned int ibatch, unsigned int crop_width, unsigned int crop_height, cudaStream_t stream){

    if(crop_width > host->width || crop_height > host->height){
        std::fprintf(stderr, "Failed to copy, invalid crop size %d x %d is larger than %d x %d\n", crop_width, crop_height, host->width, host->height);
        return;
    }

    if(crop_width > gpu->width || crop_height > gpu->height){
        std::fprintf(stderr, "Failed to copy, invalid crop size %d x %d is larger than %d x %d\n", crop_width, crop_height, gpu->width, gpu->height);
        return;
    }

    if(ibatch >= gpu->batch){
        std::fprintf(stderr, "Invalid ibatch %d is larger than %d, index out of range.\n", ibatch, gpu->batch);
        return;
    }
    if(host->format == YUVFormat::YUV422Packed_YUYV_PitchLinear){
        if(gpu->format != YUVFormat::YUV422Packed_YUYV_PitchLinear){
            /* std::fprintf(stderr, "Copied images should have the same format. host is %s, gpu is %s\n", yuvformat_name(host->format), yuvformat_name(gpu->format)); */
            return;
        }
    }

    if(gpu->format == YUVFormat::NV12PitchLinear){
        checkRuntime(cudaMemcpy2DAsync((uint8_t*)gpu->luma + ibatch * gpu->stride * gpu->height,   gpu->stride, host->data,              host->stride,
            crop_width, crop_height,     cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemcpy2DAsync((uint8_t*)gpu->chroma + ibatch * gpu->stride * gpu->height / 2, gpu->stride, host->data + host->y_area, host->stride,
            crop_width, crop_height / 2, cudaMemcpyHostToDevice, stream));
    }else if(gpu->format == YUVFormat::NV12BlockLinear){
        checkRuntime(cudaMemcpy2DToArrayAsync((cudaArray_t)gpu->luma_array,   0, ibatch * gpu->height,     host->data,              host->stride,
                crop_width, crop_height,     cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemcpy2DToArrayAsync((cudaArray_t)gpu->chroma_array, 0, ibatch * gpu->height / 2, host->data + host->y_area, host->stride,
                crop_width, crop_height / 2, cudaMemcpyHostToDevice, stream));
    }else if(gpu->format == YUVFormat::YUV422Packed_YUYV_PitchLinear){
        checkRuntime(cudaMemcpy2DAsync((uint8_t*)gpu->luma + ibatch * gpu->stride * gpu->height,   gpu->stride, host->data,              host->stride,
            crop_width * 2, crop_height,     cudaMemcpyHostToDevice, stream));
    }
}

YUVGPUImage* gpuutils::create_yuv_gpu_image(int width, int height, int batch_size, YUVFormat format,void* luma,void* chroma){

    YUVGPUImage* output = new YUVGPUImage();
    output->width  = width;
    output->height = height;
    output->batch  = batch_size;
    output->format = format;
    output->stride = width;

    if(format == YUVFormat::NV12PitchLinear){
        if(luma==nullptr&&chroma==nullptr){
            checkRuntime(cudaMalloc(&output->luma,   width * height * batch_size));
            checkRuntime(cudaMalloc(&output->chroma, width * height / 2 * batch_size));
        }else{
            output->luma=luma;
            output->chroma=chroma;
        }
    }else if(format == YUVFormat::NV12BlockLinear){
        cudaChannelFormatDesc planeDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        checkRuntime(cudaMallocArray((cudaArray_t*)&output->luma_array,   &planeDesc, width, height * batch_size));
        checkRuntime(cudaMallocArray((cudaArray_t*)&output->chroma_array, &planeDesc, width, height / 2 * batch_size));

        cudaResourceDesc luma_desc = {};
        luma_desc.resType         = cudaResourceTypeArray;
        luma_desc.res.array.array = (cudaArray_t)output->luma_array;

        cudaTextureDesc texture_desc = {};
        texture_desc.filterMode = cudaFilterModePoint;
        texture_desc.readMode   = cudaReadModeElementType;
        checkRuntime(cudaCreateTextureObject((cudaTextureObject_t*)&output->luma, &luma_desc, &texture_desc, NULL));

        cudaResourceDesc chroma_desc = {};
        chroma_desc.resType         = cudaResourceTypeArray;
        chroma_desc.res.array.array = (cudaArray_t)output->chroma_array;
        checkRuntime(cudaCreateTextureObject((cudaTextureObject_t*)&output->chroma, &chroma_desc, &texture_desc, NULL));
    }
    else if(format == YUVFormat::YUV422Packed_YUYV_PitchLinear){
        output->stride = ALIGN(width * 2);
        if(luma==nullptr){
            checkRuntime(cudaMalloc(&output->luma, output->stride * height * batch_size));
        }else{
            output->luma=luma;
        }
        
    }
    return output;
}


RGBGPUImage* gpuutils::create_rgb_gpu_image(int width, int height, int batch, PixelLayout layout, netaos::gpu::DataType dtype, void* data){

    RGBGPUImage* output = new RGBGPUImage();
    int channel = 3;
    if(layout == PixelLayout::NCHW16_BGR || layout == PixelLayout::NCHW16_RGB)
        channel = 16;

    if(layout == PixelLayout::NHWC_BGR || layout == PixelLayout::NHWC_RGB)
        output->stride = channel * width;
    else
        output->stride = 0;

    auto bytes = width * height * channel * batch * dtype_sizeof(dtype);
    /* checkRuntime(cudaMalloc(&output->data, bytes)); */
    if(data != nullptr){
        output->data = data;
    }else{
        checkRuntime(cudaMalloc(&output->data,   bytes));
    }
    output->width   = width;
    output->height  = height;
    output->batch   = batch;
    output->channel = channel;
    output->layout  = layout;
    output->dtype   = dtype;
    return output;
}

void gpuutils::batched_convert_yuv_to_rgb(
        YUVGPUImage* input, RGBGPUImage* output,
        int scaled_width, int scaled_height,
        int output_xoffset, int output_yoffset, FillColor fillcolor,
        float mean0,  float mean1,  float mean2,
        float scale0, float scale1, float scale2,
        Interpolation interp,
        cudaStream_t stream
){
    batched_convert_yuv_to_rgb(
            input->luma, input->chroma, input->width, input->stride, input->height, input->batch, input->format,
            scaled_width, scaled_height, output_xoffset, output_yoffset, fillcolor,
            output->data, output->width, output->stride, output->height, output->dtype, output->layout, interp,
            mean0, mean1, mean2, scale0, scale1, scale2, stream
            );
}
