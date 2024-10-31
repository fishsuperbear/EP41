/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_EPSFR02INFO_H
#define ARA_VEHICLE_IMPL_TYPE_EPSFR02INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct EPSFr02Info {
    ::Float eps_drive_steering_torque;
    ::UInt8 eps_inhibit_code;
    ::UInt8 eps_avail_sts;
    ::UInt8 eps_drive_steering_torque_valid;
    ::UInt8 eps_hands_on_off_state;
    ::UInt8 eps_fail_sts;
    ::UInt8 eps_interfer_dect;
    ::UInt8 eps_interfer_dect_validity;
    ::UInt8 eps_tot_torq_valid;
    ::Float eps_tot_torq;
    ::Float eps_str_col_tq;
    ::UInt8 eps_toi_fault;
    ::UInt8 eps_toi_act;
    ::UInt8 eps_toi_unavailable;
    ::UInt8 eps_fr02_msg_counter;
    ::UInt8 eps_fr02_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(eps_drive_steering_torque);
        fun(eps_inhibit_code);
        fun(eps_avail_sts);
        fun(eps_drive_steering_torque_valid);
        fun(eps_hands_on_off_state);
        fun(eps_fail_sts);
        fun(eps_interfer_dect);
        fun(eps_interfer_dect_validity);
        fun(eps_tot_torq_valid);
        fun(eps_tot_torq);
        fun(eps_str_col_tq);
        fun(eps_toi_fault);
        fun(eps_toi_act);
        fun(eps_toi_unavailable);
        fun(eps_fr02_msg_counter);
        fun(eps_fr02_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(eps_drive_steering_torque);
        fun(eps_inhibit_code);
        fun(eps_avail_sts);
        fun(eps_drive_steering_torque_valid);
        fun(eps_hands_on_off_state);
        fun(eps_fail_sts);
        fun(eps_interfer_dect);
        fun(eps_interfer_dect_validity);
        fun(eps_tot_torq_valid);
        fun(eps_tot_torq);
        fun(eps_str_col_tq);
        fun(eps_toi_fault);
        fun(eps_toi_act);
        fun(eps_toi_unavailable);
        fun(eps_fr02_msg_counter);
        fun(eps_fr02_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("eps_drive_steering_torque", eps_drive_steering_torque);
        fun("eps_inhibit_code", eps_inhibit_code);
        fun("eps_avail_sts", eps_avail_sts);
        fun("eps_drive_steering_torque_valid", eps_drive_steering_torque_valid);
        fun("eps_hands_on_off_state", eps_hands_on_off_state);
        fun("eps_fail_sts", eps_fail_sts);
        fun("eps_interfer_dect", eps_interfer_dect);
        fun("eps_interfer_dect_validity", eps_interfer_dect_validity);
        fun("eps_tot_torq_valid", eps_tot_torq_valid);
        fun("eps_tot_torq", eps_tot_torq);
        fun("eps_str_col_tq", eps_str_col_tq);
        fun("eps_toi_fault", eps_toi_fault);
        fun("eps_toi_act", eps_toi_act);
        fun("eps_toi_unavailable", eps_toi_unavailable);
        fun("eps_fr02_msg_counter", eps_fr02_msg_counter);
        fun("eps_fr02_checksum", eps_fr02_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("eps_drive_steering_torque", eps_drive_steering_torque);
        fun("eps_inhibit_code", eps_inhibit_code);
        fun("eps_avail_sts", eps_avail_sts);
        fun("eps_drive_steering_torque_valid", eps_drive_steering_torque_valid);
        fun("eps_hands_on_off_state", eps_hands_on_off_state);
        fun("eps_fail_sts", eps_fail_sts);
        fun("eps_interfer_dect", eps_interfer_dect);
        fun("eps_interfer_dect_validity", eps_interfer_dect_validity);
        fun("eps_tot_torq_valid", eps_tot_torq_valid);
        fun("eps_tot_torq", eps_tot_torq);
        fun("eps_str_col_tq", eps_str_col_tq);
        fun("eps_toi_fault", eps_toi_fault);
        fun("eps_toi_act", eps_toi_act);
        fun("eps_toi_unavailable", eps_toi_unavailable);
        fun("eps_fr02_msg_counter", eps_fr02_msg_counter);
        fun("eps_fr02_checksum", eps_fr02_checksum);
    }

    bool operator==(const ::ara::vehicle::EPSFr02Info& t) const
    {
        return (fabs(static_cast<double>(eps_drive_steering_torque - t.eps_drive_steering_torque)) < DBL_EPSILON) && (eps_inhibit_code == t.eps_inhibit_code) && (eps_avail_sts == t.eps_avail_sts) && (eps_drive_steering_torque_valid == t.eps_drive_steering_torque_valid) && (eps_hands_on_off_state == t.eps_hands_on_off_state) && (eps_fail_sts == t.eps_fail_sts) && (eps_interfer_dect == t.eps_interfer_dect) && (eps_interfer_dect_validity == t.eps_interfer_dect_validity) && (eps_tot_torq_valid == t.eps_tot_torq_valid) && (fabs(static_cast<double>(eps_tot_torq - t.eps_tot_torq)) < DBL_EPSILON) && (fabs(static_cast<double>(eps_str_col_tq - t.eps_str_col_tq)) < DBL_EPSILON) && (eps_toi_fault == t.eps_toi_fault) && (eps_toi_act == t.eps_toi_act) && (eps_toi_unavailable == t.eps_toi_unavailable) && (eps_fr02_msg_counter == t.eps_fr02_msg_counter) && (eps_fr02_checksum == t.eps_fr02_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_EPSFR02INFO_H
