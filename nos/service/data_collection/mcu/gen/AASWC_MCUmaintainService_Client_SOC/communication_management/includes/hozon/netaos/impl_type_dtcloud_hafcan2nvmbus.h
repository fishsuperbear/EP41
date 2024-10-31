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
 * @file impl_type_dtcloud_hafcan2nvmbus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFCAN2NVMBUS_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFCAN2NVMBUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_HafCAN2NVMBus {
    std::uint8_t ADCS8_RCTA_State;
    std::uint8_t ADCS8_FCTA_State;
    std::uint8_t ADCS8_DOWState;
    std::uint8_t ADCS8_RCW_State;
    std::uint8_t ADCS8_LCAState;
    std::uint8_t ADCS8_TSRState;
    std::uint8_t ADCS8_TSR_OverspeedOnOffSet;
    std::uint8_t ADCS8_ADAS_IHBCStat;
    std::uint8_t CDCS5_ResetAllSetup;
    std::uint8_t CDCS5_FactoryReset;
    std::uint8_t ADCS8_VoiceMode;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_HafCAN2NVMBus,ADCS8_RCTA_State,ADCS8_FCTA_State,ADCS8_DOWState,ADCS8_RCW_State,ADCS8_LCAState,ADCS8_TSRState,ADCS8_TSR_OverspeedOnOffSet,ADCS8_ADAS_IHBCStat,CDCS5_ResetAllSetup,CDCS5_FactoryReset,ADCS8_VoiceMode);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFCAN2NVMBUS_H_
/* EOF */