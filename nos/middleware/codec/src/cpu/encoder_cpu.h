/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef ENCODER_CPU_H_
#define DECODER_CPU_H_
#pragma once

#include "codec/include/encoder.h"

extern "C" {
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>

#include "libavutil/dict.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
}

namespace hozon {
namespace netaos {
namespace codec {

class EncoderCpu : public Encoder {
 public:
    EncoderCpu() = default;
    ~EncoderCpu();
    /**
     * @brief 初始化decode
     *
     * @return true 初始化成功返回
     * @return false 初始化失败
     */
    CodecErrc Init(const std::string& config_file);
    CodecErrc Init(const EncodeInitParam& init_param);

    /**
     * @brief 解码H265数据并输出mat
     *
     * @param in_message 编码的h265 frame
     * @param out_image 输出的解码后mat数据
     * @return true 解码成功返回
     * @return false 解码失败返回
     */
    CodecErrc Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff, FrameType& frame_type);
    CodecErrc Process(const std::string& in_buff, std::string& out_buff, FrameType& frame_type);

 private:
    AVFrame* convert_jpeg_to_yuv420p(const uint8_t* jpegData, int jpegDataLength, int width, int height);
    CodecErrc Init();

    void* context_;
    /**
     * @brief 解析h265 packet数据
     *
     * @param frame 接收的每帧数据
     * @param out_buff 解码后的数据
     */
    void avframeToYuv420P(const AVFrame* frame, std::vector<std::uint8_t>& out_buff);

    /**
     * @brief 解析h265 packet数据
     *
     * @param frame 接收的每帧数据
     * @param out_buff 解码后的数据
     */
    void avframeToNv12(const AVFrame* frame, std::string& out_buff);

    /**
     * @brief 解码器进行解码h265数据
     *
     * @param dec_ctx allocate video codec context
     * @param frame 数据帧
     * @param pkt 接收packet数据的指针
     * @param out_image 解码后的mat数据
     * @return true 解码成功返回
     * @return false 解码失败
     */
    const AVCodec* codec_;
    AVCodecParserContext* parser_;
    AVCodecContext* codeccontext_ = nullptr;
    AVFrame* frame_;
    const uint8_t* data_;
    int ret_;
    AVPacket* packet_;
    AVPixelFormat format;
    AVDictionary* param;
    std::string config_file_;
    EncodeInitParam init_param_;
    const int width = 3840;
    const int height = 2160;
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif