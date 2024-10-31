/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLRFR02INFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLRFR02INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace vehicle {
struct FLRFr02Info {
    ::UInt8 flr_acc_system_state;
    ::UInt8 flr_acc_mode;
    ::UInt8 flr_acc_sys_failure;
    ::UInt8 flr_acc_set_speed;
    ::UInt8 flr_acc_target_validity;
    ::UInt8 flr_acc_sys_info;
    ::UInt8 flr_acc_set_distance;
    ::UInt8 flr_ihu_snd_ctrl_acc_activt_reqt;
    ::UInt8 scc_obj_status;
    ::Float scc_obj_rel_spd;
    ::UInt8 flr_fr02_msg_counter;
    ::UInt8 flr_fr02_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(flr_acc_system_state);
        fun(flr_acc_mode);
        fun(flr_acc_sys_failure);
        fun(flr_acc_set_speed);
        fun(flr_acc_target_validity);
        fun(flr_acc_sys_info);
        fun(flr_acc_set_distance);
        fun(flr_ihu_snd_ctrl_acc_activt_reqt);
        fun(scc_obj_status);
        fun(scc_obj_rel_spd);
        fun(flr_fr02_msg_counter);
        fun(flr_fr02_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(flr_acc_system_state);
        fun(flr_acc_mode);
        fun(flr_acc_sys_failure);
        fun(flr_acc_set_speed);
        fun(flr_acc_target_validity);
        fun(flr_acc_sys_info);
        fun(flr_acc_set_distance);
        fun(flr_ihu_snd_ctrl_acc_activt_reqt);
        fun(scc_obj_status);
        fun(scc_obj_rel_spd);
        fun(flr_fr02_msg_counter);
        fun(flr_fr02_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("flr_acc_system_state", flr_acc_system_state);
        fun("flr_acc_mode", flr_acc_mode);
        fun("flr_acc_sys_failure", flr_acc_sys_failure);
        fun("flr_acc_set_speed", flr_acc_set_speed);
        fun("flr_acc_target_validity", flr_acc_target_validity);
        fun("flr_acc_sys_info", flr_acc_sys_info);
        fun("flr_acc_set_distance", flr_acc_set_distance);
        fun("flr_ihu_snd_ctrl_acc_activt_reqt", flr_ihu_snd_ctrl_acc_activt_reqt);
        fun("scc_obj_status", scc_obj_status);
        fun("scc_obj_rel_spd", scc_obj_rel_spd);
        fun("flr_fr02_msg_counter", flr_fr02_msg_counter);
        fun("flr_fr02_checksum", flr_fr02_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("flr_acc_system_state", flr_acc_system_state);
        fun("flr_acc_mode", flr_acc_mode);
        fun("flr_acc_sys_failure", flr_acc_sys_failure);
        fun("flr_acc_set_speed", flr_acc_set_speed);
        fun("flr_acc_target_validity", flr_acc_target_validity);
        fun("flr_acc_sys_info", flr_acc_sys_info);
        fun("flr_acc_set_distance", flr_acc_set_distance);
        fun("flr_ihu_snd_ctrl_acc_activt_reqt", flr_ihu_snd_ctrl_acc_activt_reqt);
        fun("scc_obj_status", scc_obj_status);
        fun("scc_obj_rel_spd", scc_obj_rel_spd);
        fun("flr_fr02_msg_counter", flr_fr02_msg_counter);
        fun("flr_fr02_checksum", flr_fr02_checksum);
    }

    bool operator==(const ::ara::vehicle::FLRFr02Info& t) const
    {
        return (flr_acc_system_state == t.flr_acc_system_state) && (flr_acc_mode == t.flr_acc_mode) && (flr_acc_sys_failure == t.flr_acc_sys_failure) && (flr_acc_set_speed == t.flr_acc_set_speed) && (flr_acc_target_validity == t.flr_acc_target_validity) && (flr_acc_sys_info == t.flr_acc_sys_info) && (flr_acc_set_distance == t.flr_acc_set_distance) && (flr_ihu_snd_ctrl_acc_activt_reqt == t.flr_ihu_snd_ctrl_acc_activt_reqt) && (scc_obj_status == t.scc_obj_status) && (fabs(static_cast<double>(scc_obj_rel_spd - t.scc_obj_rel_spd)) < DBL_EPSILON) && (flr_fr02_msg_counter == t.flr_fr02_msg_counter) && (flr_fr02_checksum == t.flr_fr02_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLRFR02INFO_H
