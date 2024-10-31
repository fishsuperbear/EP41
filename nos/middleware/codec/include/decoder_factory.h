/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef DECODER_FACTORY_H_
#define DECODER_FACTORY_H_
#pragma once

#include <unordered_map>
#include "codec/include/codec_def.h"
#include "codec/include/decoder.h"

namespace hozon {
namespace netaos {
namespace codec {

// 工厂类
class DecoderFactory {
 public:
  // static std::unique_ptr<Decoder> Create(std::unordered_map<std::string, std::string> config);
  static std::unique_ptr<Decoder> Create(uint32_t device_type = kDeviceType_Auto, uint32_t channel_id = 0xffu);
};

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif