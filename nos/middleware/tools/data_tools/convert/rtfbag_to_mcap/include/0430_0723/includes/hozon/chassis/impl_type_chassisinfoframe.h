/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_CHASSISINFOFRAME_H
#define HOZON_CHASSIS_IMPL_TYPE_CHASSISINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/chassis/impl_type_vcuinfo.h"
#include "hozon/chassis/impl_type_steeringinfo.h"
#include "hozon/chassis/impl_type_wheelinfo.h"
#include "hozon/chassis/impl_type_escdrivinginfo.h"
#include "hozon/chassis/impl_type_bodystateinfo.h"
#include "hozon/chassis/impl_type_centerconsoleinfo.h"
#include "hozon/chassis/impl_type_swswitchinfo.h"
#include "hozon/chassis/impl_type_avmpdsinfo.h"
#include "hozon/chassis/impl_type_algfaultdidinfo.h"
#include "hozon/chassis/impl_type_algigst.h"
#include "hozon/chassis/impl_type_algchassistime.h"
#include "hozon/chassis/impl_type_algwarnninghmiinfo.h"
#include "impl_type_uint8.h"
#include "hozon/chassis/impl_type_parkinfo.h"

namespace hozon {
namespace chassis {
struct ChassisInfoFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::chassis::VcuInfo vcu_info;
    ::hozon::chassis::SteeringInfo steering_info;
    ::hozon::chassis::WheelInfo wheel_info;
    ::hozon::chassis::EscDrivingInfo esc_driving_info;
    ::hozon::chassis::BodyStateInfo body_state_info;
    ::hozon::chassis::CenterConsoleInfo center_console_info;
    ::hozon::chassis::SWSwitchInfo swswitch_info;
    ::hozon::chassis::AvmPdsInfo avm_pds_info;
    ::hozon::chassis::AlgFaultDidInfo fault_did_info;
    ::hozon::chassis::AlgIgSt ig_status_info;
    ::hozon::chassis::AlgChassisTime chassis_time_info;
    ::hozon::chassis::AlgWarnningHmiInfo warnning_hmi_info;
    ::UInt8 error_code;
    ::hozon::chassis::ParkInfo park_info;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(vcu_info);
        fun(steering_info);
        fun(wheel_info);
        fun(esc_driving_info);
        fun(body_state_info);
        fun(center_console_info);
        fun(swswitch_info);
        fun(avm_pds_info);
        fun(fault_did_info);
        fun(ig_status_info);
        fun(chassis_time_info);
        fun(warnning_hmi_info);
        fun(error_code);
        fun(park_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(vcu_info);
        fun(steering_info);
        fun(wheel_info);
        fun(esc_driving_info);
        fun(body_state_info);
        fun(center_console_info);
        fun(swswitch_info);
        fun(avm_pds_info);
        fun(fault_did_info);
        fun(ig_status_info);
        fun(chassis_time_info);
        fun(warnning_hmi_info);
        fun(error_code);
        fun(park_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("vcu_info", vcu_info);
        fun("steering_info", steering_info);
        fun("wheel_info", wheel_info);
        fun("esc_driving_info", esc_driving_info);
        fun("body_state_info", body_state_info);
        fun("center_console_info", center_console_info);
        fun("swswitch_info", swswitch_info);
        fun("avm_pds_info", avm_pds_info);
        fun("fault_did_info", fault_did_info);
        fun("ig_status_info", ig_status_info);
        fun("chassis_time_info", chassis_time_info);
        fun("warnning_hmi_info", warnning_hmi_info);
        fun("error_code", error_code);
        fun("park_info", park_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("vcu_info", vcu_info);
        fun("steering_info", steering_info);
        fun("wheel_info", wheel_info);
        fun("esc_driving_info", esc_driving_info);
        fun("body_state_info", body_state_info);
        fun("center_console_info", center_console_info);
        fun("swswitch_info", swswitch_info);
        fun("avm_pds_info", avm_pds_info);
        fun("fault_did_info", fault_did_info);
        fun("ig_status_info", ig_status_info);
        fun("chassis_time_info", chassis_time_info);
        fun("warnning_hmi_info", warnning_hmi_info);
        fun("error_code", error_code);
        fun("park_info", park_info);
    }

    bool operator==(const ::hozon::chassis::ChassisInfoFrame& t) const
    {
        return (header == t.header) && (vcu_info == t.vcu_info) && (steering_info == t.steering_info) && (wheel_info == t.wheel_info) && (esc_driving_info == t.esc_driving_info) && (body_state_info == t.body_state_info) && (center_console_info == t.center_console_info) && (swswitch_info == t.swswitch_info) && (avm_pds_info == t.avm_pds_info) && (fault_did_info == t.fault_did_info) && (ig_status_info == t.ig_status_info) && (chassis_time_info == t.chassis_time_info) && (warnning_hmi_info == t.warnning_hmi_info) && (error_code == t.error_code) && (park_info == t.park_info);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_CHASSISINFOFRAME_H
