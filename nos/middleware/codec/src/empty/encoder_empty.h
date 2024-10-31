/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CODEC_ENCODER_EMPTY_H_
#define CODEC_ENCODER_EMPTY_H_
#pragma once

#include "codec/include/encoder.h"

namespace hozon {
namespace netaos {
namespace codec {

class EncoderEmpty : public Encoder {
 public:
  EncoderEmpty() = default;
  virtual ~EncoderEmpty() {};
  /**
   * @brief 初始化decode
   *
   * @return true 初始化成功返回
   * @return false 初始化失败
   */
  CodecErrc Init(const std::string& config_file);

  /**
   * @brief 解码H265数据并输出mat
   *
   * @param in_message yuv420sp(nv12)
   * @param out_image 输出的编码后H265流数据
   * @return 0 编码成功返回
   * @return <0> 编码失败返回
   */
  CodecErrc Process(const std::vector<std::uint8_t>& in_buff,
               std::vector<std::uint8_t>& out_buff, FrameType& frame_type);
  CodecErrc Process(const std::string& in_buff,
               std::string& out_buff, FrameType& frame_type);

 private:

};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif