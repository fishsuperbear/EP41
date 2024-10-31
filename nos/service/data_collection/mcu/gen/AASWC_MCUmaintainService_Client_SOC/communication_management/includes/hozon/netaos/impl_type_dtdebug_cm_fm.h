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
 * @file impl_type_dtdebug_cm_fm.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_FM_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_FM_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtdebug_cm_busoffstatus.h"
#include "hozon/netaos/impl_type_dtdebug_cm_e2estatus.h"
#include "hozon/netaos/impl_type_dtdebug_cm_eculoststatus.h"
#include "hozon/netaos/impl_type_dtservcallfail.h"
namespace hozon {
namespace netaos {
struct DtDebug_CM_FM {
    ::hozon::netaos::DtDebug_CM_EcuLostStatus CM_EcuLostStatus;
    ::hozon::netaos::DtDebug_CM_BusoffStatus CM_BusoffStatus;
    ::hozon::netaos::DtDebug_CM_E2EStatus CM_E2EStatus;
    ::hozon::netaos::DtServCallFail CM_CSFailStatus;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_CM_FM,CM_EcuLostStatus,CM_BusoffStatus,CM_E2EStatus,CM_CSFailStatus);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_FM_H_
/* EOF */