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
 * @file impl_type_mcudebugdatatype.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MCUDEBUGDATATYPE_H_
#define HOZON_NETAOS_IMPL_TYPE_MCUDEBUGDATATYPE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtdebug_cm_fm.h"
#include "hozon/netaos/impl_type_dtdebug_eth.h"
#include "hozon/netaos/impl_type_dtdebug_fm.h"
#include "hozon/netaos/impl_type_dtservcallfail.h"
namespace hozon {
namespace netaos {
struct MCUDebugDataType {
    ::hozon::netaos::DtDebug_CM_FM DebugData_CM;
    ::hozon::netaos::DtDebug_FM DebugData_FM;
    ::hozon::netaos::DtDebug_ETH DebugData_ETH;
    ::hozon::netaos::DtServCallFail DebugData_WDGMC0;
    ::hozon::netaos::DtServCallFail DebugData_WDGMC1;
    ::hozon::netaos::DtServCallFail DebugData_WDGMC2;
    ::hozon::netaos::DtServCallFail DebugData_WDGMC3;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MCUDebugDataType,DebugData_CM,DebugData_FM,DebugData_ETH,DebugData_WDGMC0,DebugData_WDGMC1,DebugData_WDGMC2,DebugData_WDGMC3);

#endif // HOZON_NETAOS_IMPL_TYPE_MCUDEBUGDATATYPE_H_
/* EOF */