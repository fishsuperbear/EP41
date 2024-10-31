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
 * @file impl_type_dtdebug_haflidarfailed.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_HAFLIDARFAILED_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_HAFLIDARFAILED_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtdebug_haflidarpercpalgofailed.h"
#include "hozon/netaos/impl_type_dtdebug_haflidarpercpalgointrfailed.h"
#include "hozon/netaos/impl_type_dtdebug_haflidarpercpcalibfailed.h"
#include "hozon/netaos/impl_type_dtdebug_haflidarpercpdatafailed.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtDebug_HafLidarFailed {
    std::uint8_t Ins_data_Failed;
    ::hozon::netaos::DtDebug_HafLidarPercpAlgoFailed Lidar_Percp_Algo_Failed;
    ::hozon::netaos::DtDebug_HafLidarPercpDataFailed Lidar_Percp_Data_Failed;
    ::hozon::netaos::DtDebug_HafLidarPercpAlgoIntrFailed Lidar_Percp_Algo_Intr_Failed;
    ::hozon::netaos::DtDebug_HafLidarPercpCalibFailed Lidar_Percp_Calib_Failed;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_HafLidarFailed,Ins_data_Failed,Lidar_Percp_Algo_Failed,Lidar_Percp_Data_Failed,Lidar_Percp_Algo_Intr_Failed,Lidar_Percp_Calib_Failed);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_HAFLIDARFAILED_H_
/* EOF */