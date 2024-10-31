/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef DECODER_H_
#define DECODER_H_
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "codec/include/codec_def.h"
#include "codec/include/codec_error_domain.h"

namespace hozon {
namespace netaos {
namespace codec {

/**
 * @brief orin 场景下，通过回调的方式获得解码结果，保存在指针中，指针类型：NvSciBufObj
 *
 * @return >-1 返回成功
 * @return -1 返回失败
 */
using pfnCbDecoderNvMediaOutput = std::function<int32_t(const DecoderBufNvSpecific)>;

struct DecoderNvMediaCb {
    pfnCbDecoderNvMediaOutput CbDecoderNvMediaOutput;
};

class Decoder {
   public:
    /**
     * @brief 初始化decode
     *
     * @return 0 初始化成功返回
     * @return -1 初始化失败
     */
    virtual CodecErrc Init(const std::string& config_file) { return kDecodeNotImplemented; }

    virtual CodecErrc Init(const DecodeInitParam& init_param) { return kDecodeNotImplemented; }

    virtual CodecErrc Init(DecoderNvMediaCb decoderNvMediaCb) { return kDecodeNotImplemented; }

    virtual CodecErrc Init(const PicInfos& pic_infos) { return kDecodeNotImplemented; }

    /**
     * @brief 解码H265数据并输出mat
     *
     * @param in_message 编码的h265 frame
     * @param out_image 输出的解码后mat数据
     * @return 0 解码成功返回
     * @return -1 解码失败返回
     */

    virtual CodecErrc Process(const std::vector<std::uint8_t>& in_buff) { return kDecodeNotImplemented; }

    virtual CodecErrc Process(const std::string& in_buff) { return kDecodeNotImplemented; }

    virtual CodecErrc Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff) { return kDecodeNotImplemented; }

    virtual CodecErrc Process(const std::string& in_buff, std::string& out_buff) { return kDecodeNotImplemented; };

    virtual CodecErrc Process(const std::vector<std::uint8_t>& in_buff, DecoderBufNvSpecific& out_buff) { return kDecodeNotImplemented; }

    virtual CodecErrc Process(const std::string& in_buff, DecoderBufNvSpecific& out_buff) { return kDecodeNotImplemented; }

    virtual CodecErrc Process(const DecodeBufInfo& info, const std::string& in_buff) { return kDecodeNotImplemented; }

    virtual CodecErrc Process(const std::string& in_buff, void* out_buff, int32_t* len) { return kDecodeNotImplemented; };

    /**
     * @brief 获取解码后的width
     *
     * @return >-1 返回width
     * @return -1 获取width失败
     */
    virtual int32_t GetWidth() { return 0; }

    /**
     * @brief 获取解码后的height
     *
     * @return >-1 返回height
     * @return -1 获取height失败
     */
    virtual int32_t GetHeight() { return 0; }

    /**
     * @brief 获取解码后的格式，格式类型使用ffmpeg的AVPixelFormat定义
     *
     * @return >-1 返回格式
     * @return -1 获取格式失败
     */
    virtual int32_t GetFormat() { return 0; }

    virtual int32_t GetStride() { return -1; }

    virtual ~Decoder() {}
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif