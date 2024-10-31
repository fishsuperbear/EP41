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
 * @file impl_type_algbodystateinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGBODYSTATEINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGBODYSTATEINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgBodyStateInfo {
    std::uint8_t BCM_FLDrOpn;
    std::uint8_t BCM_FRDrOpn;
    std::uint8_t BCM_RLDrOpn;
    std::uint8_t BCM_RRDrOpn;
    std::uint8_t BCM_TGOpn;
    bool BCM_HodOpen;
    std::uint8_t BCM_DrvSeatbeltBucklesta;
    std::uint8_t BCM_FrontWiperSt;
    std::uint8_t BCM_FrontWiperWorkSts;
    std::uint8_t BCM_HighBeamSt;
    std::uint8_t CS1_HighBeamReqSt;
    std::uint8_t BCM_LowBeamSt;
    std::uint8_t HazardLampSt;
    bool BCM_FrontFogLampSt;
    bool BCM_RearFogLampSt;
    bool BCM_LeftTurnLightSt;
    bool BCM_RightTurnLightSt;
    std::uint8_t BCM_TurnLightSW;
    std::uint8_t BCM_FrontLampSt;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgBodyStateInfo,BCM_FLDrOpn,BCM_FRDrOpn,BCM_RLDrOpn,BCM_RRDrOpn,BCM_TGOpn,BCM_HodOpen,BCM_DrvSeatbeltBucklesta,BCM_FrontWiperSt,BCM_FrontWiperWorkSts,BCM_HighBeamSt,CS1_HighBeamReqSt,BCM_LowBeamSt,HazardLampSt,BCM_FrontFogLampSt,BCM_RearFogLampSt,BCM_LeftTurnLightSt,BCM_RightTurnLightSt,BCM_TurnLightSW,BCM_FrontLampSt);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGBODYSTATEINFO_H_
/* EOF */