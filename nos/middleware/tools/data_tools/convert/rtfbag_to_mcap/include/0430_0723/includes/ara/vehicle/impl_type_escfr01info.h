/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_ESCFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_ESCFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct ESCFr01Info {
    ::UInt8 esc_disable;
    ::UInt8 esc_active;
    ::UInt8 esc_fail;
    ::UInt8 ebd_fail;
    ::UInt8 abs_fail;
    ::UInt8 abs_active;
    ::UInt8 tcs_active;
    ::UInt8 esc_off_lamp_req;
    ::UInt8 tcs_fail;
    ::UInt8 tcs_disable;
    ::UInt8 esc_dec_ctrl_ena;
    ::UInt8 esc_apa_stand_still;
    ::UInt8 esc_apa_sts;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(esc_disable);
        fun(esc_active);
        fun(esc_fail);
        fun(ebd_fail);
        fun(abs_fail);
        fun(abs_active);
        fun(tcs_active);
        fun(esc_off_lamp_req);
        fun(tcs_fail);
        fun(tcs_disable);
        fun(esc_dec_ctrl_ena);
        fun(esc_apa_stand_still);
        fun(esc_apa_sts);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(esc_disable);
        fun(esc_active);
        fun(esc_fail);
        fun(ebd_fail);
        fun(abs_fail);
        fun(abs_active);
        fun(tcs_active);
        fun(esc_off_lamp_req);
        fun(tcs_fail);
        fun(tcs_disable);
        fun(esc_dec_ctrl_ena);
        fun(esc_apa_stand_still);
        fun(esc_apa_sts);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("esc_disable", esc_disable);
        fun("esc_active", esc_active);
        fun("esc_fail", esc_fail);
        fun("ebd_fail", ebd_fail);
        fun("abs_fail", abs_fail);
        fun("abs_active", abs_active);
        fun("tcs_active", tcs_active);
        fun("esc_off_lamp_req", esc_off_lamp_req);
        fun("tcs_fail", tcs_fail);
        fun("tcs_disable", tcs_disable);
        fun("esc_dec_ctrl_ena", esc_dec_ctrl_ena);
        fun("esc_apa_stand_still", esc_apa_stand_still);
        fun("esc_apa_sts", esc_apa_sts);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("esc_disable", esc_disable);
        fun("esc_active", esc_active);
        fun("esc_fail", esc_fail);
        fun("ebd_fail", ebd_fail);
        fun("abs_fail", abs_fail);
        fun("abs_active", abs_active);
        fun("tcs_active", tcs_active);
        fun("esc_off_lamp_req", esc_off_lamp_req);
        fun("tcs_fail", tcs_fail);
        fun("tcs_disable", tcs_disable);
        fun("esc_dec_ctrl_ena", esc_dec_ctrl_ena);
        fun("esc_apa_stand_still", esc_apa_stand_still);
        fun("esc_apa_sts", esc_apa_sts);
    }

    bool operator==(const ::ara::vehicle::ESCFr01Info& t) const
    {
        return (esc_disable == t.esc_disable) && (esc_active == t.esc_active) && (esc_fail == t.esc_fail) && (ebd_fail == t.ebd_fail) && (abs_fail == t.abs_fail) && (abs_active == t.abs_active) && (tcs_active == t.tcs_active) && (esc_off_lamp_req == t.esc_off_lamp_req) && (tcs_fail == t.tcs_fail) && (tcs_disable == t.tcs_disable) && (esc_dec_ctrl_ena == t.esc_dec_ctrl_ena) && (esc_apa_stand_still == t.esc_apa_stand_still) && (esc_apa_sts == t.esc_apa_sts);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_ESCFR01INFO_H
