/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip def
 */

#ifndef DOSOMEIP_DEF_H
#define DOSOMEIP_DEF_H

#include <stdint.h>
#include <vector>
#include "diag/dosomeip/log/dosomeip_logger.h"
#include "diag/dosomeip/config/dosomeip_config.h"

namespace hozon {
namespace netaos {
namespace diag {

enum class DOSOMEIP_RESULT : uint16_t
{
    DOSOMEIP_RESULT_OK                 = 0x00,    /* Success */
    DOSOMEIP_RESULT_INVALID_SA         = 0x01,    /* Source address is invalid */
    DOSOMEIP_RESULT_UNKNOWN_TA         = 0x02,    /* Target address is unknown */
    DOSOMEIP_RESULT_UNKNOWN_SA         = 0x03,    /* Source address is unknown */
    DOSOMEIP_RESULT_INITIAL_FAILED     = 0x04,    /* Module init error */
    DOSOMEIP_RESULT_NOT_INITIALIZED    = 0x05,    /* Call interface order error */
    DOSOMEIP_RESULT_ALREADY_INITED     = 0x06,    /* Repeated initialization */
    DOSOMEIP_RESULT_PARAMETER_ERROR    = 0x07,    /* The parameter is incorrect */
    DOSOMEIP_RESULT_CONFIG_ERROR       = 0x08,    /* load config failed */
    DOSOMEIP_RESULT_TIMEOUT_A          = 0x09,    /* Communication timeout */
    DOSOMEIP_RESULT_ERROR              = 0xFF     /* Common error */
};

enum class TargetAddressType : uint16_t
{
    kPhysical = 0x00,
    kFunctional = 0x01
};

struct DoSomeIPReqUdsMessage {
    DoSomeIPReqUdsMessage():udsSa(0), udsTa(0), taType(TargetAddressType::kFunctional){}
    uint16_t udsSa;
    uint16_t udsTa;
    TargetAddressType taType;
    std::vector<uint8_t> udsData;
};

struct DoSomeIPRespUdsMessage {
    DoSomeIPRespUdsMessage():udsSa(0), udsTa(0), result(0), taType(TargetAddressType::kFunctional){}
    uint16_t udsSa;
    uint16_t udsTa;
    uint32_t result;
    TargetAddressType taType;
    std::vector<uint8_t> udsData;
};

enum class DOSOMEIP_REGISTER_STATUS : uint16_t
{
    DOSOMEIP_REGISTER_STATUS_OK         = 0x00,    
    DOSOMEIP_REGISTER_STATUS_TIMEOUT    = 0x01     
};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DOSOMEIP_DEF_H 