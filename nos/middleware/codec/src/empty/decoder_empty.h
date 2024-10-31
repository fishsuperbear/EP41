/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef DECODER_EMPTY_H_
#define DECODER_EMPTY_H_
#pragma once
#include "codec/include/decoder.h"

namespace hozon {
namespace netaos {
namespace codec {

class DecoderEmpty : public Decoder {
   public:
    DecoderEmpty() = default;
    ~DecoderEmpty(){};
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
    CodecErrc Process(const std::string& in_buff, std::string& out_buff);
    int GetWidth();
    int GetHeight();
    int GetFormat();
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif