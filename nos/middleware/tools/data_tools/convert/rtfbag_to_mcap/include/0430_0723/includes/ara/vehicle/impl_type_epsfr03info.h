/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_EPSFR03INFO_H
#define ARA_VEHICLE_IMPL_TYPE_EPSFR03INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"

namespace ara {
namespace vehicle {
struct EPSFr03Info {
    ::UInt8 eps_steering_angle_spd_valid;
    ::UInt8 eps_steering_angle_valid;
    ::UInt8 eps_calibrated_status;
    ::UInt16 eps_steering_angle_spd;
    ::Float eps_steering_angle;
    ::UInt8 eps_fr03_msg_counter;
    ::UInt8 eps_fr03_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(eps_steering_angle_spd_valid);
        fun(eps_steering_angle_valid);
        fun(eps_calibrated_status);
        fun(eps_steering_angle_spd);
        fun(eps_steering_angle);
        fun(eps_fr03_msg_counter);
        fun(eps_fr03_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(eps_steering_angle_spd_valid);
        fun(eps_steering_angle_valid);
        fun(eps_calibrated_status);
        fun(eps_steering_angle_spd);
        fun(eps_steering_angle);
        fun(eps_fr03_msg_counter);
        fun(eps_fr03_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("eps_steering_angle_spd_valid", eps_steering_angle_spd_valid);
        fun("eps_steering_angle_valid", eps_steering_angle_valid);
        fun("eps_calibrated_status", eps_calibrated_status);
        fun("eps_steering_angle_spd", eps_steering_angle_spd);
        fun("eps_steering_angle", eps_steering_angle);
        fun("eps_fr03_msg_counter", eps_fr03_msg_counter);
        fun("eps_fr03_checksum", eps_fr03_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("eps_steering_angle_spd_valid", eps_steering_angle_spd_valid);
        fun("eps_steering_angle_valid", eps_steering_angle_valid);
        fun("eps_calibrated_status", eps_calibrated_status);
        fun("eps_steering_angle_spd", eps_steering_angle_spd);
        fun("eps_steering_angle", eps_steering_angle);
        fun("eps_fr03_msg_counter", eps_fr03_msg_counter);
        fun("eps_fr03_checksum", eps_fr03_checksum);
    }

    bool operator==(const ::ara::vehicle::EPSFr03Info& t) const
    {
        return (eps_steering_angle_spd_valid == t.eps_steering_angle_spd_valid) && (eps_steering_angle_valid == t.eps_steering_angle_valid) && (eps_calibrated_status == t.eps_calibrated_status) && (eps_steering_angle_spd == t.eps_steering_angle_spd) && (fabs(static_cast<double>(eps_steering_angle - t.eps_steering_angle)) < DBL_EPSILON) && (eps_fr03_msg_counter == t.eps_fr03_msg_counter) && (eps_fr03_checksum == t.eps_fr03_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_EPSFR03INFO_H
