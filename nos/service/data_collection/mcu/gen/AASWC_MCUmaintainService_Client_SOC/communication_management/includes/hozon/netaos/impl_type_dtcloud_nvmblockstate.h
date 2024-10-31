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
 * @file impl_type_dtcloud_nvmblockstate.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_NVMBLOCKSTATE_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_NVMBLOCKSTATE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_NVMBlockState {
    std::uint8_t FBL_BootData_State;
    std::uint8_t SecOc_State;
    std::uint8_t BSWConfig_State;
    std::uint8_t System_ResetWakeup_State;
    std::uint8_t System_Awake_State;
    std::uint8_t ASW_RemmberState_State;
    std::uint8_t System_Cfg0_State;
    std::uint8_t System_Cfg1_State;
    std::uint8_t DTC_Information0_State;
    std::uint8_t DTC_Information1_State;
    std::uint8_t DTC_Information2_State;
    std::uint8_t DTC_Information3_State;
    std::uint8_t Reserved0_State;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_NVMBlockState,FBL_BootData_State,SecOc_State,BSWConfig_State,System_ResetWakeup_State,System_Awake_State,ASW_RemmberState_State,System_Cfg0_State,System_Cfg1_State,DTC_Information0_State,DTC_Information1_State,DTC_Information2_State,DTC_Information3_State,Reserved0_State);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_NVMBLOCKSTATE_H_
/* EOF */