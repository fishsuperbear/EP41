/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CODEC_ENCODER_H_
#define CODEC_ENCODER_H_
#pragma once

#include <memory>
#include <vector>
#include "codec/include/codec_error_domain.h"
#include "codec/include/codec_def.h"

namespace hozon {
namespace netaos {
namespace codec {

class Encoder {
 public:
  /**
   * @brief 初始化encode
   *
   * @return 0 初始化成功返回
   * @return -1 初始化失败
   */
  virtual CodecErrc Init(const std::string& config_file) { return kDecodeNotImplemented; };
  virtual CodecErrc Init(const EncodeInitParam& init_param) { return kDecodeNotImplemented; };

  /**
   * @brief 编码H265数据并输出mat
   *
   * @param in_message 编码的h265 frame
   * @param out_image 输出的解码后mat数据
   * @return 0 编码成功返回
   * @return <0 编码失败返回
   */
  virtual CodecErrc Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff, FrameType& frame_type) { return kEncodeNotImplemented; };
  virtual CodecErrc Process(const std::string& in_buff, std::string& out_buff, FrameType& frame_type) { return kEncodeNotImplemented; };
  virtual CodecErrc Process(const void *in, std::vector<uint8_t>& out_buff, FrameType& frame_type) { return kEncodeNotImplemented; };
  virtual CodecErrc Process(const void *in, std::string& out_buff, FrameType& frame_type) { return kEncodeNotImplemented; };

  virtual CodecErrc GetEncoderParam(int param_type, void **param) { return kEncodeNotImplemented; }
  virtual CodecErrc SetEncoderParam(int param_type, void *param) { return kEncodeNotImplemented; }

  virtual ~Encoder() {};
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif