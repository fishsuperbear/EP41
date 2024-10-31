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
 * @file impl_type_dtcloud_cm_runnablestatus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_RUNNABLESTATUS_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_RUNNABLESTATUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_CM_RunnableStatus {
    std::uint32_t CM_Core0_MainFunction_Cnt;
    std::uint32_t CM_Core0_MainFunction_FM_Cnt;
    std::uint32_t CM_Core0_MainFunctionTx_Cnt;
    std::uint32_t CM_Core1_MainFunctionRx_Cnt;
    std::uint32_t CM_Core1_MainFunctionTx_Cnt;
    std::uint32_t CM_Core3_MainFunctionRx_Cnt;
    std::uint32_t CM_Core3_MainFunctionTx_Cnt;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_CM_RunnableStatus,CM_Core0_MainFunction_Cnt,CM_Core0_MainFunction_FM_Cnt,CM_Core0_MainFunctionTx_Cnt,CM_Core1_MainFunctionRx_Cnt,CM_Core1_MainFunctionTx_Cnt,CM_Core3_MainFunctionRx_Cnt,CM_Core3_MainFunctionTx_Cnt);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_RUNNABLESTATUS_H_
/* EOF */