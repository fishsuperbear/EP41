/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-10-16 16:43:35
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-10-25 17:19:04
 * @FilePath: /nos/middleware/codec/src/encoder_factory.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "codec/include/encoder_factory.h"

#include <iostream>

#ifdef PLAT_ORIN
#include "codec/src/cpu/encoder_cpu.h"
#include "codec/src/nvmedia/encoder_nvmedia.h"
#include "codec/src/nvmedia/encoder_nvmedia_v2.h"
#endif

#ifdef PLAT_X86
#include "codec/src/cpu/encoder_cpu.h"
#endif

#ifdef PLAT_MDC
// #include "codec/src/mdc/xxx.h"
#endif

#include "codec/src/codec_logger.h"
#include "codec/src/empty/encoder_empty.h"

namespace hozon {
namespace netaos {
namespace codec {

// std::unique_ptr<Encoder> EncoderFactory::Create(std::unordered_map<std::string, std::string> config) {
//     CodecLogger::GetInstance().setLogLevel(static_cast<int32_t>(CodecLogger::CodecLogLevelType::INFO));
//     CodecLogger::GetInstance().CreateLogger("CODEC");
// #ifdef PLAT_MDC
//     return std::make_unique<EncoderEmpty>();
// #else
// #ifdef PLAT_ORIN
//     return std::make_unique<EncoderNvmediaV2>();
//     // return std::make_unique<EncoderCpu>();

// #else
//     return std::make_unique<EncoderCpu>();
// #endif
// #endif
//     return std::make_unique<EncoderEmpty>();
// }

std::unique_ptr<Encoder> EncoderFactory::Create(uint32_t device_type, uint32_t channel_id) {
    CodecLogger::GetInstance().setLogLevel(static_cast<int32_t>(CodecLogger::CodecLogLevelType::INFO));
    CodecLogger::GetInstance().CreateLogger("CODEC");

#ifdef PLAT_MDC
    return std::make_unique<EncoderEmpty>();
#else
#ifdef PLAT_ORIN

    std::unique_ptr<Encoder> encoder;

    switch (device_type) {
        case kDeviceType_Auto:
            encoder = std::make_unique<EncoderNvmediaV2>();
            break;
        case kDeviceType_Cpu:
            encoder = std::make_unique<EncoderCpu>();
            break;
        case kDeviceType_NvMedia:
            encoder = std::make_unique<EncoderNvmediaV2>();
            break;
        case kDeviceType_NvMedia_NvStream:
            // Not implemented yet.
            break;
    }

    return encoder;
#else
    return std::make_unique<EncoderCpu>();

#endif
#endif
    return std::make_unique<EncoderEmpty>();
}

}  // namespace codec
}  // namespace netaos
}  // namespace hozon