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
 * @file impl_type_algegowarninginfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGEGOWARNINGINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGEGOWARNINGINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgEgoWarningInfo {
    std::uint8_t ADCS8_LCARightWarnSt;
    std::uint8_t ADCS8_LCALeftWarnSt;
    std::uint8_t ADCS8_LCASystemFaultStatus;
    std::uint8_t ADCS8_LCAState;
    std::uint8_t ADCS8_DOW_State;
    std::uint8_t ADCS12_DOWWarnAudioplay;
    std::uint8_t ADCS8_DOWLeftWarnState;
    std::uint8_t ADCS8_DOWRightWarnState;
    std::uint8_t ADCS8_DOW_SystemFaultStatus;
    std::uint8_t ADCS8_RCTA_State;
    std::uint8_t ADCS12_RCTAWarnAudioplay;
    std::uint8_t ADCS8_RCTAWarnObjType;
    std::uint8_t ADCS8_RCTALeftWarnSt;
    std::uint8_t ADCS8_RCTARightWarnSt;
    std::uint8_t ADCS8_RCTA_SystemFaultStatus;
    std::uint8_t ADCS8_FCTA_State;
    std::uint8_t ADCS12_FCTAWarnAudioplay;
    std::uint8_t ADCS8_FCTAWarnObjType;
    std::uint8_t ADCS8_FCTALeftActiveSt;
    std::uint8_t ADCS8_FCTARightActiveSt;
    std::uint8_t ADCS8_FCTA_SystemFaultStatus;
    std::uint8_t ADCS8_RCW_State;
    std::uint8_t ADCS12_RCWWarnAudioplay;
    std::uint8_t ADCS8_RCW_WarnState;
    std::uint8_t ADCS8_RCW_SystemFaultStatus;
    std::uint8_t ADCS8_VoiceMode;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEgoWarningInfo,ADCS8_LCARightWarnSt,ADCS8_LCALeftWarnSt,ADCS8_LCASystemFaultStatus,ADCS8_LCAState,ADCS8_DOW_State,ADCS12_DOWWarnAudioplay,ADCS8_DOWLeftWarnState,ADCS8_DOWRightWarnState,ADCS8_DOW_SystemFaultStatus,ADCS8_RCTA_State,ADCS12_RCTAWarnAudioplay,ADCS8_RCTAWarnObjType,ADCS8_RCTALeftWarnSt,ADCS8_RCTARightWarnSt,ADCS8_RCTA_SystemFaultStatus,ADCS8_FCTA_State,ADCS12_FCTAWarnAudioplay,ADCS8_FCTAWarnObjType,ADCS8_FCTALeftActiveSt,ADCS8_FCTARightActiveSt,ADCS8_FCTA_SystemFaultStatus,ADCS8_RCW_State,ADCS12_RCWWarnAudioplay,ADCS8_RCW_WarnState,ADCS8_RCW_SystemFaultStatus,ADCS8_VoiceMode);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGEGOWARNINGINFO_H_
/* EOF */