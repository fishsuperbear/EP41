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
 * @file impl_type_mcuclouddatatype.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MCUCLOUDDATATYPE_H_
#define HOZON_NETAOS_IMPL_TYPE_MCUCLOUDDATATYPE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_adas.h"
#include "hozon/netaos/impl_type_dtcloud_cm_running.h"
#include "hozon/netaos/impl_type_dtcloud_eth.h"
#include "hozon/netaos/impl_type_dtcloud_fm.h"
#include "hozon/netaos/impl_type_dtcloud_hm.h"
#include "hozon/netaos/impl_type_dtcloud_nvm.h"
#include "hozon/netaos/impl_type_dtcloud_os.h"
#include "hozon/netaos/impl_type_dtcloud_pwronoffdata.h"
#include "hozon/netaos/impl_type_dtcloud_sm.h"
#include "hozon/netaos/impl_type_mcuclouddatatype_array_1000.h"
namespace hozon {
namespace netaos {
struct MCUCloudDataType {
    ::hozon::netaos::DtCloud_CM_Running CMData;
    ::hozon::netaos::DtCloud_HM HMData;
    ::hozon::netaos::DtCloud_FM FMData;
    ::hozon::netaos::DtCloud_SM SMData;
    ::hozon::netaos::DtCloud_NVM NVMData;
    ::hozon::netaos::DtCloud_ETH ETHData;
    ::hozon::netaos::DtCloud_OS OSData;
    ::hozon::netaos::MCUCloudDataType_Array_1000 MCUReserve;
    ::hozon::netaos::DtCloud_PwrOnOffData PwrOnOffData;
    ::hozon::netaos::DtCloud_ADAS ADASData;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MCUCloudDataType,CMData,HMData,FMData,SMData,NVMData,ETHData,OSData,MCUReserve,PwrOnOffData,ADASData);

#endif // HOZON_NETAOS_IMPL_TYPE_MCUCLOUDDATATYPE_H_
/* EOF */