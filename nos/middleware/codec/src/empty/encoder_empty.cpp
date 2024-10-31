/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/

#include "encoder_empty.h"
#include "codec/include/codec_error_domain.h"
#include "codec/src/codec_logger.h"

namespace hozon {
namespace netaos {
namespace codec {

CodecErrc EncoderEmpty::Init(const std::string& config_file) {
    return kEncodeInitError;
}

CodecErrc EncoderEmpty::EncoderEmpty::Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff, FrameType& frame_type) {
    return kEncodeFailed;
}

CodecErrc EncoderEmpty::Process(const std::string& in_buff, std::string& out_buff, FrameType& frame_type) {
    return kEncodeFailed;
}

}
}
}