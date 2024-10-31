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
 * @file impl_type_algmcuegonnpmsg.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGMCUEGONNPMSG_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGMCUEGONNPMSG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgMcuEgoNNPMsg {
    std::uint8_t LongitudCtrlDecToStopReq;
    std::uint8_t LongitudCtrlDriveOff;
    std::uint8_t DriveOffinhibition;
    std::uint8_t DriveOffinhibitionObjType;
    std::uint8_t Lcsndconfirm;
    std::uint8_t TurnLightReqSt;
    std::uint8_t Lcsndrequest;
    std::uint8_t PayModeConfirm;
    std::uint8_t SpdAdaptComfirm;
    std::uint8_t ALC_mode;
    std::uint8_t ADSDriving_mode;
    std::uint8_t longitudCtrlSetSpeed;
    std::uint8_t longitudCtrlSetDistance;
    std::uint8_t LowBeamSt;
    std::uint8_t HighBeamSt;
    std::uint8_t HazardLampSt;
    std::uint8_t LowHighBeamSt;
    std::uint8_t HornSt;
    std::uint8_t NNPSysState;
    std::uint8_t acc_target_id;
    std::uint8_t alc_warnning_target_id;
    std::uint8_t alc_warnning_state;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgMcuEgoNNPMsg,LongitudCtrlDecToStopReq,LongitudCtrlDriveOff,DriveOffinhibition,DriveOffinhibitionObjType,Lcsndconfirm,TurnLightReqSt,Lcsndrequest,PayModeConfirm,SpdAdaptComfirm,ALC_mode,ADSDriving_mode,longitudCtrlSetSpeed,longitudCtrlSetDistance,LowBeamSt,HighBeamSt,HazardLampSt,LowHighBeamSt,HornSt,NNPSysState,acc_target_id,alc_warnning_target_id,alc_warnning_state);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGMCUEGONNPMSG_H_
/* EOF */