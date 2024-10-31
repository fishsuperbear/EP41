/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CODEC_ENCODER_FACTORY_H_
#define CODEC_ENCODER_FACTORY_H_
#pragma once

#include <unordered_map>
#include "codec/include/codec_def.h"
#include "codec/include/encoder.h"

namespace hozon {
namespace netaos {
namespace codec {

// 工厂类
class EncoderFactory {
 public:
  // static std::unique_ptr<Encoder> Create(std::unordered_map<std::string, std::string> config);
  static std::unique_ptr<Encoder> Create(uint32_t device_type = kDeviceType_Auto, uint32_t channel_id = 0xffu);
};

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif