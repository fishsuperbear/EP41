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
 * @file impl_type_algwheelinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGWHEELINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGWHEELINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgWheelInfo {
    float ESC_FLWheelSpeed;
    bool ESC_FLWheelSpeedValid;
    std::uint8_t ESC_FLWheelDirection;
    float ESC_FRWheelSpeed;
    bool ESC_FRWheelSpeedValid;
    std::uint8_t ESC_FRWheelDirection;
    float ESC_RLWheelSpeed;
    bool ESC_RLWheelSpeedValid;
    std::uint8_t ESC_RLWheelDirection;
    float ESC_RRWheelSpeed;
    bool ESC_RRWheelSpeedValid;
    std::uint8_t ESC_RRWheelDirection;
    float ESC_FL_WhlPulCnt;
    float ESC_FR_WhlPulCnt;
    float ESC_RL_WhlPulCnt;
    float ESC_RR_WhlPulCnt;
    bool ESC_FL_WhlPulCntValid;
    bool ESC_FR_WhlPulCntValid;
    bool ESC_RL_WhlPulCntValid;
    bool ESC_RR_WhlPulCntValid;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgWheelInfo,ESC_FLWheelSpeed,ESC_FLWheelSpeedValid,ESC_FLWheelDirection,ESC_FRWheelSpeed,ESC_FRWheelSpeedValid,ESC_FRWheelDirection,ESC_RLWheelSpeed,ESC_RLWheelSpeedValid,ESC_RLWheelDirection,ESC_RRWheelSpeed,ESC_RRWheelSpeedValid,ESC_RRWheelDirection,ESC_FL_WhlPulCnt,ESC_FR_WhlPulCnt,ESC_RL_WhlPulCnt,ESC_RR_WhlPulCnt,ESC_FL_WhlPulCntValid,ESC_FR_WhlPulCntValid,ESC_RL_WhlPulCntValid,ESC_RR_WhlPulCntValid);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGWHEELINFO_H_
/* EOF */