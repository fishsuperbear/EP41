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
 * @file impl_type_canfd_msgfe.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSGFE_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSGFE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_MsgFE {
    std::uint8_t ADCS4_AVM_DayNightStatus;
    std::uint8_t ADCS4_AVM_Sts;
    std::uint8_t ADCS4_RPA_RemoteParkFinishReq;
    std::uint8_t ADCS4_PA_failinfo;
    std::uint8_t ADCS4_HPA_failinfo;
    std::uint8_t ADCS4_AVM_SysSoundIndication;
    std::uint8_t ADCS4_APA_FunctionMode;
    std::uint8_t ADCS4_Text;
    std::uint8_t ADCS4_ParkingswithReq;
    std::uint8_t ADCS4_Slotavaliable;
    std::uint8_t ADCS4_HPA_deleteMapSt;
    std::uint8_t ADCS4_HPA_uploadMapSt;
    std::uint16_t ADCS4_ParkingTime;
    std::uint8_t ADCS4_HPAWarningInfo;
    std::uint8_t ADCS4_AVM_vedioReq;
    std::uint8_t ADCS4_HPA_FunctionMode;
    std::uint8_t ADCS4_NNS_FunctionMode;
    std::uint8_t ADCS4_TractionswithReq;
    std::uint8_t ADCS4_RPA_FunctionMode;
    std::uint8_t ADCS4_GMWarnState;
    std::uint8_t ADCS4_GMState;
    std::uint8_t ADCS4_locationSt;
    std::uint8_t ADCS4_GMWorkState;
    std::uint8_t ADCS4_DCWworkSt;
    std::uint8_t ADCS4_DCWposition;
    std::uint8_t ADCS4_DCWlevel;
    std::uint8_t ADCS4_DCWtext;
    std::uint8_t ADCS4_GMS_text;
    std::uint8_t ADCS4_GMS_Failinfo;
    std::uint8_t ADCS4_TBA_Distance;
    std::uint8_t ADCS4_FindCarAvmStatus;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_MsgFE,ADCS4_AVM_DayNightStatus,ADCS4_AVM_Sts,ADCS4_RPA_RemoteParkFinishReq,ADCS4_PA_failinfo,ADCS4_HPA_failinfo,ADCS4_AVM_SysSoundIndication,ADCS4_APA_FunctionMode,ADCS4_Text,ADCS4_ParkingswithReq,ADCS4_Slotavaliable,ADCS4_HPA_deleteMapSt,ADCS4_HPA_uploadMapSt,ADCS4_ParkingTime,ADCS4_HPAWarningInfo,ADCS4_AVM_vedioReq,ADCS4_HPA_FunctionMode,ADCS4_NNS_FunctionMode,ADCS4_TractionswithReq,ADCS4_RPA_FunctionMode,ADCS4_GMWarnState,ADCS4_GMState,ADCS4_locationSt,ADCS4_GMWorkState,ADCS4_DCWworkSt,ADCS4_DCWposition,ADCS4_DCWlevel,ADCS4_DCWtext,ADCS4_GMS_text,ADCS4_GMS_Failinfo,ADCS4_TBA_Distance,ADCS4_FindCarAvmStatus);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSGFE_H_
/* EOF */