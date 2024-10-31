/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_PDCIFO_AVM_H
#define HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_PDCIFO_AVM_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"

namespace hozon {
namespace sensors {
struct UssRawData_PdcIfo_AVM {
    ::UInt8 flc_distance;
    ::UInt8 flm_distance;
    ::UInt8 frc_distance;
    ::UInt8 frm_distance;
    ::UInt8 rlc_distance;
    ::UInt8 rlm_distance;
    ::UInt8 rrm_distance;
    ::UInt8 rrc_distance;
    ::UInt16 fls_distance;
    ::UInt16 frs_distance;
    ::UInt16 rls_distance;
    ::UInt16 rrs_distance;
    ::UInt8 flc_fault_status;
    ::UInt8 frm_fault_status;
    ::UInt8 frc_fault_status;
    ::UInt8 rlc_fault_status;
    ::UInt8 rlm_fault_status;
    ::UInt8 rrm_fault_status;
    ::UInt8 rrc_fault_status;
    ::UInt8 flm_fault_status;
    ::UInt8 fls_fault_status;
    ::UInt8 frs_fault_status;
    ::UInt8 rls_fault_status;
    ::UInt8 rrs_fault_status;
    ::UInt8 ls_distance;
    ::UInt8 rs_distance;
    ::Float fpa_min_dist;
    ::Float rpa_min_dist;
    ::Float pa_obstacles_mindist;
    ::UInt8 pa_obstacles_area;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(flc_distance);
        fun(flm_distance);
        fun(frc_distance);
        fun(frm_distance);
        fun(rlc_distance);
        fun(rlm_distance);
        fun(rrm_distance);
        fun(rrc_distance);
        fun(fls_distance);
        fun(frs_distance);
        fun(rls_distance);
        fun(rrs_distance);
        fun(flc_fault_status);
        fun(frm_fault_status);
        fun(frc_fault_status);
        fun(rlc_fault_status);
        fun(rlm_fault_status);
        fun(rrm_fault_status);
        fun(rrc_fault_status);
        fun(flm_fault_status);
        fun(fls_fault_status);
        fun(frs_fault_status);
        fun(rls_fault_status);
        fun(rrs_fault_status);
        fun(ls_distance);
        fun(rs_distance);
        fun(fpa_min_dist);
        fun(rpa_min_dist);
        fun(pa_obstacles_mindist);
        fun(pa_obstacles_area);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(flc_distance);
        fun(flm_distance);
        fun(frc_distance);
        fun(frm_distance);
        fun(rlc_distance);
        fun(rlm_distance);
        fun(rrm_distance);
        fun(rrc_distance);
        fun(fls_distance);
        fun(frs_distance);
        fun(rls_distance);
        fun(rrs_distance);
        fun(flc_fault_status);
        fun(frm_fault_status);
        fun(frc_fault_status);
        fun(rlc_fault_status);
        fun(rlm_fault_status);
        fun(rrm_fault_status);
        fun(rrc_fault_status);
        fun(flm_fault_status);
        fun(fls_fault_status);
        fun(frs_fault_status);
        fun(rls_fault_status);
        fun(rrs_fault_status);
        fun(ls_distance);
        fun(rs_distance);
        fun(fpa_min_dist);
        fun(rpa_min_dist);
        fun(pa_obstacles_mindist);
        fun(pa_obstacles_area);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("flc_distance", flc_distance);
        fun("flm_distance", flm_distance);
        fun("frc_distance", frc_distance);
        fun("frm_distance", frm_distance);
        fun("rlc_distance", rlc_distance);
        fun("rlm_distance", rlm_distance);
        fun("rrm_distance", rrm_distance);
        fun("rrc_distance", rrc_distance);
        fun("fls_distance", fls_distance);
        fun("frs_distance", frs_distance);
        fun("rls_distance", rls_distance);
        fun("rrs_distance", rrs_distance);
        fun("flc_fault_status", flc_fault_status);
        fun("frm_fault_status", frm_fault_status);
        fun("frc_fault_status", frc_fault_status);
        fun("rlc_fault_status", rlc_fault_status);
        fun("rlm_fault_status", rlm_fault_status);
        fun("rrm_fault_status", rrm_fault_status);
        fun("rrc_fault_status", rrc_fault_status);
        fun("flm_fault_status", flm_fault_status);
        fun("fls_fault_status", fls_fault_status);
        fun("frs_fault_status", frs_fault_status);
        fun("rls_fault_status", rls_fault_status);
        fun("rrs_fault_status", rrs_fault_status);
        fun("ls_distance", ls_distance);
        fun("rs_distance", rs_distance);
        fun("fpa_min_dist", fpa_min_dist);
        fun("rpa_min_dist", rpa_min_dist);
        fun("pa_obstacles_mindist", pa_obstacles_mindist);
        fun("pa_obstacles_area", pa_obstacles_area);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("flc_distance", flc_distance);
        fun("flm_distance", flm_distance);
        fun("frc_distance", frc_distance);
        fun("frm_distance", frm_distance);
        fun("rlc_distance", rlc_distance);
        fun("rlm_distance", rlm_distance);
        fun("rrm_distance", rrm_distance);
        fun("rrc_distance", rrc_distance);
        fun("fls_distance", fls_distance);
        fun("frs_distance", frs_distance);
        fun("rls_distance", rls_distance);
        fun("rrs_distance", rrs_distance);
        fun("flc_fault_status", flc_fault_status);
        fun("frm_fault_status", frm_fault_status);
        fun("frc_fault_status", frc_fault_status);
        fun("rlc_fault_status", rlc_fault_status);
        fun("rlm_fault_status", rlm_fault_status);
        fun("rrm_fault_status", rrm_fault_status);
        fun("rrc_fault_status", rrc_fault_status);
        fun("flm_fault_status", flm_fault_status);
        fun("fls_fault_status", fls_fault_status);
        fun("frs_fault_status", frs_fault_status);
        fun("rls_fault_status", rls_fault_status);
        fun("rrs_fault_status", rrs_fault_status);
        fun("ls_distance", ls_distance);
        fun("rs_distance", rs_distance);
        fun("fpa_min_dist", fpa_min_dist);
        fun("rpa_min_dist", rpa_min_dist);
        fun("pa_obstacles_mindist", pa_obstacles_mindist);
        fun("pa_obstacles_area", pa_obstacles_area);
    }

    bool operator==(const ::hozon::sensors::UssRawData_PdcIfo_AVM& t) const
    {
        return (flc_distance == t.flc_distance) && (flm_distance == t.flm_distance) && (frc_distance == t.frc_distance) && (frm_distance == t.frm_distance) && (rlc_distance == t.rlc_distance) && (rlm_distance == t.rlm_distance) && (rrm_distance == t.rrm_distance) && (rrc_distance == t.rrc_distance) && (fls_distance == t.fls_distance) && (frs_distance == t.frs_distance) && (rls_distance == t.rls_distance) && (rrs_distance == t.rrs_distance) && (flc_fault_status == t.flc_fault_status) && (frm_fault_status == t.frm_fault_status) && (frc_fault_status == t.frc_fault_status) && (rlc_fault_status == t.rlc_fault_status) && (rlm_fault_status == t.rlm_fault_status) && (rrm_fault_status == t.rrm_fault_status) && (rrc_fault_status == t.rrc_fault_status) && (flm_fault_status == t.flm_fault_status) && (fls_fault_status == t.fls_fault_status) && (frs_fault_status == t.frs_fault_status) && (rls_fault_status == t.rls_fault_status) && (rrs_fault_status == t.rrs_fault_status) && (ls_distance == t.ls_distance) && (rs_distance == t.rs_distance) && (fabs(static_cast<double>(fpa_min_dist - t.fpa_min_dist)) < DBL_EPSILON) && (fabs(static_cast<double>(rpa_min_dist - t.rpa_min_dist)) < DBL_EPSILON) && (fabs(static_cast<double>(pa_obstacles_mindist - t.pa_obstacles_mindist)) < DBL_EPSILON) && (pa_obstacles_area == t.pa_obstacles_area);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_USSRAWDATA_PDCIFO_AVM_H
