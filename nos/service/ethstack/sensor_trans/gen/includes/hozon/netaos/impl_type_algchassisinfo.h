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
 * @file impl_type_algchassisinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGCHASSISINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGCHASSISINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_algavmpdsinfo.h"
#include "hozon/netaos/impl_type_algbodystateinfo.h"
#include "hozon/netaos/impl_type_algcenterconsoleinfo.h"
#include "hozon/netaos/impl_type_algchassistime.h"
#include "hozon/netaos/impl_type_algescdrivinginfo.h"
#include "hozon/netaos/impl_type_algfaultdidinfo.h"
#include "hozon/netaos/impl_type_algigst.h"
#include "hozon/netaos/impl_type_algparkinfo.h"
#include "hozon/netaos/impl_type_algsteeringinfo.h"
#include "hozon/netaos/impl_type_algswswitchinfo.h"
#include "hozon/netaos/impl_type_algvcuinfo.h"
#include "hozon/netaos/impl_type_algwarnninghmiinfo.h"
#include "hozon/netaos/impl_type_algwheelinfo.h"
#include "hozon/netaos/impl_type_hafheader.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgChassisInfo {
    ::hozon::netaos::HafHeader header;
    bool isValid;
    ::hozon::netaos::AlgVcuInfo vcu_info;
    ::hozon::netaos::AlgSteeringInfo steering_info;
    ::hozon::netaos::AlgWheelInfo wheel_info;
    ::hozon::netaos::AlgEscDrivingInfo esc_driving_info;
    ::hozon::netaos::AlgBodyStateInfo body_state_info;
    ::hozon::netaos::AlgCenterConsoleInfo center_console_info;
    ::hozon::netaos::AlgParkInfo park_info;
    ::hozon::netaos::AlgSWSwitchInfo swswitch_info;
    ::hozon::netaos::AlgAvmPdsInfo avm_pds_info;
    ::hozon::netaos::AlgFaultDidInfo fault_did_info;
    ::hozon::netaos::AlgIgSt ig_status_info;
    ::hozon::netaos::AlgChassisTime chassis_time_info;
    ::hozon::netaos::AlgWarnningHmiInfo warnning_hmi_info;
    std::uint8_t error_code;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgChassisInfo,header,isValid,vcu_info,steering_info,wheel_info,esc_driving_info,body_state_info,center_console_info,park_info,swswitch_info,avm_pds_info,fault_did_info,ig_status_info,chassis_time_info,warnning_hmi_info,error_code);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGCHASSISINFO_H_
/* EOF */