/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-10-16 16:43:35
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-10-27 11:11:42
 * @FilePath: /nos/middleware/codec/src/decoder_factory.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "codec/include/decoder_factory.h"

#ifdef PLAT_ORIN
#include "codec/src/cpu/decoder_cpu.h"
#include "codec/src/cuda/decoder_cuda.h"
#include "codec/src/nvmedia/decoder_nvmedia.h"
#include "codec/src/nvmedia/decoder_nvstream.h"
#endif

#ifdef PLAT_X86
#include "codec/src/cpu/decoder_cpu.h"
#include "codec/src/cuda/decoder_cuda.h"
#endif

#ifdef PLAT_MDC
// #include "codec/src/mdc/xxx.h"
#endif

#include "codec/src/codec_logger.h"
#include "codec/src/empty/decoder_empty.h"

namespace hozon {
namespace netaos {
namespace codec {

// std::unique_ptr<Decoder> codec::DecoderFactory::Create(std::unordered_map<std::string, std::string> config) {

// #ifdef PLAT_MDC
//     return std::make_unique<DecoderEmpty>();
// #endif

// #ifdef PLAT_ORIN
//     return std::make_unique<DecoderNvStream>();
// #endif

// #ifdef PLAT_X86
//     return std::make_unique<DecoderCpu>();
// #endif

//     return std::make_unique<DecoderEmpty>();
// }

std::unique_ptr<Decoder> codec::DecoderFactory::Create(uint32_t device_type, uint32_t channel_id) {
    CodecLogger::GetInstance().setLogLevel(static_cast<int32_t>(CodecLogger::CodecLogLevelType::INFO));
    CodecLogger::GetInstance().CreateLogger("CODEC");

#ifdef PLAT_MDC
    return std::make_unique<DecoderEmpty>();
#endif

#ifdef PLAT_ORIN

    std::unique_ptr<Decoder> decoder;

    switch (device_type) {
        case kDeviceType_Auto:
            decoder = std::make_unique<DecoderNvMedia>();
            break;
        case kDeviceType_Cpu:
            decoder = std::make_unique<DecoderCpu>();
            break;
        case kDeviceType_NvMedia:
            decoder = std::make_unique<DecoderNvMedia>();
            break;
        case kDeviceType_NvMedia_NvStream:
            decoder = std::make_unique<DecoderNvStream>();
            break;
    }

    return decoder;

#endif

#ifdef PLAT_X86
    std::unique_ptr<Decoder> decoder;
    switch (device_type) {
        case kDeviceType_Auto:
        case kDeviceType_Cpu:
            decoder = std::make_unique<DecoderCpu>();
            break;
        case kDeviceType_Cuda:
            decoder = std::make_unique<DecoderCuda>();
            break;
        default:
            decoder = std::make_unique<DecoderCpu>();
            break;
    }
    return decoder;

#endif

    return std::make_unique<DecoderEmpty>();
}

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
