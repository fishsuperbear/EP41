/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef DECODER_NVMEDIA_H_
#define DECODER_NVMEDIA_H_
#pragma once

#include <atomic>
#include "codec/include/decoder.h"
#include "codec/src/nvmedia/CUtils.hpp"
#include "codec/src/nvmedia/ide/video_dec_ctx.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

namespace hozon {
namespace netaos {
namespace codec {

class DecoderNvMedia : public Decoder {
   public:
    DecoderNvMedia();
    ~DecoderNvMedia();
    /**
     * @brief 初始化decode
     *
     * @return true 初始化成功返回
     * @return false 初始化失败
     */
    CodecErrc Init(const std::string& config_file);
    CodecErrc Init(const DecodeInitParam& init_param);
    CodecErrc Init(DecoderNvMediaCb decoderNvMediaCb);

    /**
     * @brief 解码H265数据并输出mat
     *
     * @param in_message 编码的h265 frame
     * @param out_image 输出的解码后mat数据
     * @return true 解码成功返回
     * @return false 解码失败返回
     */
    CodecErrc Process(const std::vector<std::uint8_t>& in_buff);
    CodecErrc Process(const std::string& in_buff);
    CodecErrc Process(const std::vector<std::uint8_t>& in_buff, DecoderBufNvSpecific& out_buff);
    CodecErrc Process(const std::string& in_buff, DecoderBufNvSpecific& out_buff);
    int GetWidth();
    int GetHeight();
    int GetFormat();

    bool MapNvScibuf2Cuda(NvSciBufObj scibuf, uint8_t idx);
    bool BlToPlConvert(uint32_t packetIndex, void* dstptr);

    DecoderNvMediaCb decoderNvMediaCb_;
    DecoderBufNvSpecific out_buff_;

    struct CudaData {
        void* cuda_ptr[MAX_DEC_BUFFERS]{0};
        cudaExternalMemory_t ext_mem[MAX_DEC_BUFFERS]{0};
        cudaMipmappedArray_t mipmap_array[MAX_DEC_BUFFERS][3]{0};
        cudaArray_t mipLevel_array[MAX_DEC_BUFFERS][3]{0};
        BufferAttrs buf_attrs[MAX_DEC_BUFFERS]{};
    } mapping_cuda_data_;

   private:
    bool CheckVersion(void);
    CodecErrc Decode(const std::string& in_buff, DecoderBufNvSpecific& out_buff);

    // int32_t cbBeginSequence(void* ptr, const NvMediaParserSeqInfo* pnvsi);
    // NvMediaStatus cbDecodePicture(void* ptr, NvMediaParserPictureData* pd);
    // NvMediaStatus cbDisplayPicture(void* ptr, NvMediaRefSurface* p, int64_t llPts);
    // void cbUnhandledNALU(void* ptr, const uint8_t* buf, int32_t size);
    // NvMediaStatus cbAllocPictureBuffer(void* ptr, NvMediaRefSurface** p);
    // int StreamVC1SimpleProfile(VideoDecCtx ctx);
    // void cbRelease(void* ptr, NvMediaRefSurface* p);
    // void cbAddRef(void* ptr, NvMediaRefSurface* p);
    // NvMediaStatus cbGetBackwardUpdates(void* ptr, NvMediaVP9BackwardUpdates* backwardUpdate);

    cudaStream_t streamWaiter_ = nullptr;

    VideoDecCtx ctx{0};
    NvMediaParserClientCb clientCb;
    std::string OutputYUVFilename;
    std::string filename;
    std::atomic<bool> cur_frame_decoded_;
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif