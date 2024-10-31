/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef DECODER_CUDA_H_
#define DECODER_CUDA_H_
#pragma once
#include <memory>
#include <thread>
#include "codec/include/decoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#include "libavutil/dict.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
}

namespace hozon {
namespace netaos {
namespace codec {

class DecoderCuda : public Decoder {
   public:
    DecoderCuda() = default;
    ~DecoderCuda();
    /**
     * @brief 初始化decode
     *
     * @return true 初始化成功返回
     * @return false 初始化失败
     */
    CodecErrc Init(const std::string& config_file);
    CodecErrc Init(const DecodeInitParam& init_param);

    /**
     * @brief 解码H265数据并输出mat
     *
     * @param in_message 编码的h265 frame
     * @param out_image 输出的解码后mat数据
     * @return true 解码成功返回
     * @return false 解码失败返回
     */
    CodecErrc Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff);
    CodecErrc Process(const std::string& in_buff, void* out_buff, int32_t* len) override;
    int32_t GetWidth() override;
    int32_t GetHeight() override;
    int32_t GetFormat() override;
    int32_t GetStride() override;

    typedef struct {
        const AVCodec* codec = nullptr;
        AVBufferRef* hw_device_ctx = nullptr;
        AVCodecParserContext* parser = nullptr;
        AVCodecContext* codeccontext = nullptr;
        AVPixelFormat pix_format;
        AVHWDeviceType device_type = AV_HWDEVICE_TYPE_NONE;
        AVFrame* frame = nullptr;
        AVPacket* pkt = nullptr;
        int width;
        int height;
        int format;
    } AvContext;

   protected:
    static AVPixelFormat GetHwFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);
    bool DecodeToNv12(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len);
    bool DecodeToYUYV(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len);

   private:
    DecodeInitParam init_param_{0};
    AvContext av_ctx_{0};
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif