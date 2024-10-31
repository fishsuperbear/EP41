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
 * @file impl_type_dtdebug_cm_eculoststatus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_ECULOSTSTATUS_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_ECULOSTSTATUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtDebug_CM_EcuLostStatus {
    std::uint8_t CM_Ecu_Lost_ACU;
    std::uint8_t CM_Ecu_Lost_BDCS;
    std::uint8_t CM_Ecu_Lost_BTM;
    std::uint8_t CM_Ecu_Lost_CDCS;
    std::uint8_t CM_Ecu_Lost_DDCU;
    std::uint8_t CM_Ecu_Lost_EDU;
    std::uint8_t CM_Ecu_Lost_EPS;
    std::uint8_t CM_Ecu_Lost_FMCU;
    std::uint8_t CM_Ecu_Lost_GW;
    std::uint8_t CM_Ecu_Lost_ICU;
    std::uint8_t CM_Ecu_Lost_IDB;
    std::uint8_t CM_Ecu_Lost_MCU;
    std::uint8_t CM_Ecu_Lost_PDCU;
    std::uint8_t CM_Ecu_Lost_RCU;
    std::uint8_t CM_Ecu_Lost_TBOX;
    std::uint8_t CM_Ecu_Lost_PDCS;
    std::uint8_t CM_Ecu_Lost_SOC;
    std::uint8_t CM_Ecu_Lost_BMS;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_CM_EcuLostStatus,CM_Ecu_Lost_ACU,CM_Ecu_Lost_BDCS,CM_Ecu_Lost_BTM,CM_Ecu_Lost_CDCS,CM_Ecu_Lost_DDCU,CM_Ecu_Lost_EDU,CM_Ecu_Lost_EPS,CM_Ecu_Lost_FMCU,CM_Ecu_Lost_GW,CM_Ecu_Lost_ICU,CM_Ecu_Lost_IDB,CM_Ecu_Lost_MCU,CM_Ecu_Lost_PDCU,CM_Ecu_Lost_RCU,CM_Ecu_Lost_TBOX,CM_Ecu_Lost_PDCS,CM_Ecu_Lost_SOC,CM_Ecu_Lost_BMS);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_ECULOSTSTATUS_H_
/* EOF */