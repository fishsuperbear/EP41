/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file impl_type_algegotsrihbchmiinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGEGOTSRIHBCHMIINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGEGOTSRIHBCHMIINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgEgoTsrIhbcHmiInfo {
    std::uint8_t ADCS8_TSRState;
    std::uint8_t ADCS8_TSR_SpeedWarnState;
    std::uint8_t ADCS8_TSR_OverspeedOnOffSet;
    std::uint8_t ADCS8_TSR_LimitOverspeedSet;
    std::uint8_t ADCS12_SpeedWarnAudioplay;
    std::uint8_t ADCS8_TSR_SystemFaultStatus;
    std::uint8_t ADCS8_TSR_SpeedSign;
    std::uint8_t ADCS8_TSR_ForbiddenSign;
    std::uint8_t ADCS8_TSR_StrLightColor;
    std::uint8_t ADCS8_TSR_RightLightColor;
    std::uint8_t ADCS8_TSR_LeftLightColor;
    std::uint8_t ADCS2_ADAS_IHBCSysState;
    std::uint8_t ADCS8_ADAS_IHBCStat;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEgoTsrIhbcHmiInfo,ADCS8_TSRState,ADCS8_TSR_SpeedWarnState,ADCS8_TSR_OverspeedOnOffSet,ADCS8_TSR_LimitOverspeedSet,ADCS12_SpeedWarnAudioplay,ADCS8_TSR_SystemFaultStatus,ADCS8_TSR_SpeedSign,ADCS8_TSR_ForbiddenSign,ADCS8_TSR_StrLightColor,ADCS8_TSR_RightLightColor,ADCS8_TSR_LeftLightColor,ADCS2_ADAS_IHBCSysState,ADCS8_ADAS_IHBCStat);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGEGOTSRIHBCHMIINFO_H_
/* EOF */