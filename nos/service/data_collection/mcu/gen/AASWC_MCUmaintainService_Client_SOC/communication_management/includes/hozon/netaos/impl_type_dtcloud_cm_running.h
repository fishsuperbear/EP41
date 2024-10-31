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
 * @file impl_type_dtcloud_cm_running.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_RUNNING_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_RUNNING_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_cm_datatransmitstatus.h"
#include "hozon/netaos/impl_type_dtcloud_cm_runnablestatus.h"
#include "hozon/netaos/impl_type_dtcloud_cm_signalsendstatus.h"
namespace hozon {
namespace netaos {
struct DtCloud_CM_Running {
    ::hozon::netaos::DtCloud_CM_RunnableStatus CM_RunnableStatus;
    ::hozon::netaos::DtCloud_CM_DataTransmitStatus CM_DataTransmitStatus;
    ::hozon::netaos::DtCloud_CM_SignalSendStatus CM_SignalSendStatus;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_CM_Running,CM_RunnableStatus,CM_DataTransmitStatus,CM_SignalSendStatus);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_RUNNING_H_
/* EOF */