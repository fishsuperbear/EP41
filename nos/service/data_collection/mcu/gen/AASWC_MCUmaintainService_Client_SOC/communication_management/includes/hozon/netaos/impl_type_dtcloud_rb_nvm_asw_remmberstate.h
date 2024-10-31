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
 * @file impl_type_dtcloud_rb_nvm_asw_remmberstate.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_RB_NVM_ASW_REMMBERSTATE_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_RB_NVM_ASW_REMMBERSTATE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_rb_NvM_ASW_RemmberState {
    std::uint8_t LDWMode;
    std::uint8_t LKSSetspeed;
    std::uint8_t LDPMode;
    std::uint8_t VoiceMode;
    std::uint8_t ACCOnOffState;
    std::uint8_t DCLCSysState;
    std::uint8_t NNPState;
    std::uint8_t AutoOnOffSet;
    std::uint8_t ALCMode;
    std::uint8_t ADSDrivingMode;
    std::uint8_t TSR_SLFStatefeedback;
    std::uint8_t RCTA_OnOffSet;
    std::uint8_t FCTA_OnOffSet;
    std::uint8_t DOW_OnOffSet;
    std::uint8_t RCW_OnOffSet;
    std::uint8_t LCA_OnOffSet;
    std::uint8_t TSR_OnOffSet;
    std::uint8_t TSR_OverspeedOnOffSet;
    std::uint8_t IHBC_OnOffSet;
    float CtrlYawrateOffset;
    float CtrlYawOffset;
    float CtrlAxOffset;
    float CtrlSteerOffset;
    float CtrlAccelDeadzone;
    std::uint8_t ADCS8_FCWSensitiveLevel;
    std::uint8_t AEB_OnOffSet;
    std::uint8_t FCW_OnOffSet;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_rb_NvM_ASW_RemmberState,LDWMode,LKSSetspeed,LDPMode,VoiceMode,ACCOnOffState,DCLCSysState,NNPState,AutoOnOffSet,ALCMode,ADSDrivingMode,TSR_SLFStatefeedback,RCTA_OnOffSet,FCTA_OnOffSet,DOW_OnOffSet,RCW_OnOffSet,LCA_OnOffSet,TSR_OnOffSet,TSR_OverspeedOnOffSet,IHBC_OnOffSet,CtrlYawrateOffset,CtrlYawOffset,CtrlAxOffset,CtrlSteerOffset,CtrlAccelDeadzone,ADCS8_FCWSensitiveLevel,AEB_OnOffSet,FCW_OnOffSet);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_RB_NVM_ASW_REMMBERSTATE_H_
/* EOF */