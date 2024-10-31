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
 * @file impl_type_dtcloud_lifecycle_mgr_info.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_LIFECYCLE_MGR_INFO_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_LIFECYCLE_MGR_INFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_lifecycle.h"
#include "hozon/netaos/impl_type_dtcloud_socsysst.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_LifeCycle_Mgr_Info {
    std::uint8_t PM_LifeCycle_State;
    ::hozon::netaos::DtCloud_Lifecycle g_SocLifecycleInfo;
    ::hozon::netaos::DtCloud_SocSysSt g_McuState_Soc;
    std::uint8_t mcu_sys_state;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_LifeCycle_Mgr_Info,PM_LifeCycle_State,g_SocLifecycleInfo,g_McuState_Soc,mcu_sys_state);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_LIFECYCLE_MGR_INFO_H_
/* EOF */