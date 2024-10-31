/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/

#include "encoder_nvmedia.h"
#include "codec/include/codec_error_domain.h"
#include "codec/src/codec_logger.h"

namespace hozon {
namespace netaos {
namespace codec {

EncoderNvmedia::~EncoderNvmedia() {
    ::Deinit(context_);
}

CodecErrc EncoderNvmedia::Init(const std::string& config_file) {

    return static_cast<CodecErrc>(::Init(config_file.c_str(), &context_, true, nullptr));
}

CodecErrc EncoderNvmedia::EncoderNvmedia::Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff, FrameType& frame_type) {
    // TODO: optimize performance
    uint8_t* out = nullptr;
    int size = 0;
    CodecErrc res = static_cast<CodecErrc>(::Process(in_buff.data(), &out, &size, reinterpret_cast<int*>(&frame_type), context_));

    if (res == kEncodeSuccess) {
        out_buff.resize(size);
        memcpy(out_buff.data(), out, size);
    }
    free(out);
    return res;
}

CodecErrc EncoderNvmedia::Process(const std::string& in_buff, std::string& out_buff, FrameType& frame_type) {
    // TODO: optimize performance
    uint8_t* out = nullptr;
    int size = 0;
    CodecErrc res = static_cast<CodecErrc>(::Process(reinterpret_cast<uint8_t const *>(in_buff.data()), &out, &size, reinterpret_cast<int*>(&frame_type), context_));
    if (res == kEncodeSuccess) {
        out_buff.assign(reinterpret_cast<char*>(out), size);
    }

    free(out);
    return res;
}

CodecErrc EncoderNvmedia::Process(const void *in, std::vector<std::uint8_t>& out_buff, FrameType& frame_type) {
    // TODO: optimize performance
    uint8_t* out = nullptr;
    int size = 0;
    CodecErrc res = static_cast<CodecErrc>(::Process(in, &out, &size, reinterpret_cast<int*>(&frame_type), context_));
    if (res == kEncodeSuccess) {
        out_buff.resize(size);
        memcpy(out_buff.data(), out, size);
    }
    free(out);
    return res;
}

CodecErrc EncoderNvmedia::Process(const void *in, std::string& out_buff, FrameType& frame_type) {
    // TODO: optimize performance
    uint8_t* out = nullptr;
    int size = 0;
    CodecErrc res = static_cast<CodecErrc>(::Process(in, &out, &size, reinterpret_cast<int*>(&frame_type), context_));
    if (res == kEncodeSuccess) {
        out_buff.assign(reinterpret_cast<char*>(out), size);
    }
    free(out);
    return res;
}

CodecErrc EncoderNvmedia::GetEncoderParam(int param_type, void **param) {
    return static_cast<CodecErrc>(::GetEncoderParam(context_, param_type, param));
}

CodecErrc EncoderNvmedia::SetEncoderParam(int param_type, void *param) {
    return static_cast<CodecErrc>(::SetEncoderParam(context_, param_type, param));
}

}
}
}