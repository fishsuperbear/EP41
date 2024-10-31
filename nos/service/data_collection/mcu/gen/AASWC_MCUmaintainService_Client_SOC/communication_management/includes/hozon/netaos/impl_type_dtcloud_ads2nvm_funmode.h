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
 * @file impl_type_dtcloud_ads2nvm_funmode.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_ADS2NVM_FUNMODE_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_ADS2NVM_FUNMODE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_ADS2NVM_FunMode {
    std::uint8_t LDWMode;
    std::uint8_t LKSSetspeed;
    std::uint8_t LDPMode;
    std::uint8_t VoiceMode;
    std::uint8_t ACCOnOffState;
    std::uint8_t DCLCSysState;
    std::uint8_t NNP_State;
    std::uint8_t AutoOnOffSet;
    std::uint8_t ALC_mode;
    std::uint8_t ADSDriving_mode;
    std::uint8_t TSR_SLFStatefeedback;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_ADS2NVM_FunMode,LDWMode,LKSSetspeed,LDPMode,VoiceMode,ACCOnOffState,DCLCSysState,NNP_State,AutoOnOffSet,ALC_mode,ADSDriving_mode,TSR_SLFStatefeedback);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_ADS2NVM_FUNMODE_H_
/* EOF */