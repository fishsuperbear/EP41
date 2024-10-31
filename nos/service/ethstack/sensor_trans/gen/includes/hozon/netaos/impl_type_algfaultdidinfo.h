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
 * @file impl_type_algfaultdidinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGFAULTDIDINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGFAULTDIDINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgFaultDidInfo {
    bool BDCS10_AC_OutsideTempValid;
    float BDCS10_AC_OutsideTemp;
    std::uint8_t Power_Supply_Voltage;
    bool ICU1_VehicleSpdDisplayValid;
    float ICU1_VehicleSpdDisplay;
    float ICU2_Odometer;
    std::uint8_t BDCS1_PowerManageMode;
    std::uint8_t Ignition_status;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgFaultDidInfo,BDCS10_AC_OutsideTempValid,BDCS10_AC_OutsideTemp,Power_Supply_Voltage,ICU1_VehicleSpdDisplayValid,ICU1_VehicleSpdDisplay,ICU2_Odometer,BDCS1_PowerManageMode,Ignition_status);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGFAULTDIDINFO_H_
/* EOF */