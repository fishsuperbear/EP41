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
 * @file impl_type_canfd_msg8f.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG8F_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG8F_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg8F {
    std::uint8_t ADCS11_Parking_WorkSts;
    std::uint8_t ADCS11_PA_Recover;
    std::uint8_t ADCS11_TurnLampReq;
    std::uint8_t ADCS11_SetPathwayWorkSts;
    std::uint8_t ADCS11_HPAGuideSts;
    std::uint8_t ADCS11_HPA_WorkSts;
    std::uint8_t ADCS11_APA_RemoteLockCtrl;
    std::uint8_t ADCS11_APA_requestMirrorFoldSt;
    std::uint8_t ADCS11_PrkngFctnModeReq;
    std::uint8_t ADCS11_HPAPathlearning_WorkSts;
    std::uint8_t ADCS11_HPAPathwaytoCloud_WorkSts;
    std::uint8_t ADCS11_APA_RemoteStartReq;
    std::uint8_t ADCS11_APA_RemoteShutdownReq;
    std::uint8_t ADCS11_MirrorFoldrequest;
    std::uint8_t ADCS11_PA_ParkingFnMd;
    std::uint8_t ADCS11_HPA_PathlearningSt;
    std::uint8_t ADCS11_HPA_Path_exist;
    std::uint8_t ADCS11_PA_PickSt;
    std::uint8_t ADCS11_PA_StopReq;
    std::uint8_t ADCS11_HPA_Pathavailable_ID2;
    std::uint8_t ADCS11_HPA_Pathavailable_ID1;
    std::uint8_t ADCS11_Currentgear;
    std::uint16_t ADCS11_HPA_learnpathdistance;
    std::uint8_t ADCS11_PA_lampreq;
    std::uint8_t ADCS11_HPA_BacktoEntrance;
    std::uint8_t ADCS11_HPA_BacktoStart;
    std::uint8_t ADCS11_HighBeam;
    std::uint8_t ADCS11_LowBeam;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg8F,ADCS11_Parking_WorkSts,ADCS11_PA_Recover,ADCS11_TurnLampReq,ADCS11_SetPathwayWorkSts,ADCS11_HPAGuideSts,ADCS11_HPA_WorkSts,ADCS11_APA_RemoteLockCtrl,ADCS11_APA_requestMirrorFoldSt,ADCS11_PrkngFctnModeReq,ADCS11_HPAPathlearning_WorkSts,ADCS11_HPAPathwaytoCloud_WorkSts,ADCS11_APA_RemoteStartReq,ADCS11_APA_RemoteShutdownReq,ADCS11_MirrorFoldrequest,ADCS11_PA_ParkingFnMd,ADCS11_HPA_PathlearningSt,ADCS11_HPA_Path_exist,ADCS11_PA_PickSt,ADCS11_PA_StopReq,ADCS11_HPA_Pathavailable_ID2,ADCS11_HPA_Pathavailable_ID1,ADCS11_Currentgear,ADCS11_HPA_learnpathdistance,ADCS11_PA_lampreq,ADCS11_HPA_BacktoEntrance,ADCS11_HPA_BacktoStart,ADCS11_HighBeam,ADCS11_LowBeam);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG8F_H_
/* EOF */