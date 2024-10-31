/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_USSRAWDATASET_H
#define HOZON_SENSORS_IMPL_TYPE_USSRAWDATASET_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64.h"
#include "hozon/sensors/impl_type_ussrawdata_apa.h"
#include "hozon/sensors/impl_type_ussrawdata_pdc.h"
#include "impl_type_uint8.h"
#include "hozon/sensors/impl_type_ussrawdata_pdcifo_avm.h"

namespace hozon {
namespace sensors {
struct UssRawDataSet {
    ::UInt64 time_stamp;
    ::hozon::sensors::UssRawData_APA fls_info;
    ::hozon::sensors::UssRawData_APA frs_info;
    ::hozon::sensors::UssRawData_APA rls_info;
    ::hozon::sensors::UssRawData_APA rrs_info;
    ::hozon::sensors::UssRawData_PDC flc_info;
    ::hozon::sensors::UssRawData_PDC flm_info;
    ::hozon::sensors::UssRawData_PDC frm_info;
    ::hozon::sensors::UssRawData_PDC frc_info;
    ::hozon::sensors::UssRawData_PDC rlc_info;
    ::hozon::sensors::UssRawData_PDC rlm_info;
    ::hozon::sensors::UssRawData_PDC rrm_info;
    ::hozon::sensors::UssRawData_PDC rrc_info;
    ::UInt8 counter;
    ::hozon::sensors::UssRawData_PdcIfo_AVM pdcinfo_avm;
    ::hozon::sensors::UssRawData_PdcIfo_AVM pdcinfo_filter;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time_stamp);
        fun(fls_info);
        fun(frs_info);
        fun(rls_info);
        fun(rrs_info);
        fun(flc_info);
        fun(flm_info);
        fun(frm_info);
        fun(frc_info);
        fun(rlc_info);
        fun(rlm_info);
        fun(rrm_info);
        fun(rrc_info);
        fun(counter);
        fun(pdcinfo_avm);
        fun(pdcinfo_filter);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time_stamp);
        fun(fls_info);
        fun(frs_info);
        fun(rls_info);
        fun(rrs_info);
        fun(flc_info);
        fun(flm_info);
        fun(frm_info);
        fun(frc_info);
        fun(rlc_info);
        fun(rlm_info);
        fun(rrm_info);
        fun(rrc_info);
        fun(counter);
        fun(pdcinfo_avm);
        fun(pdcinfo_filter);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time_stamp", time_stamp);
        fun("fls_info", fls_info);
        fun("frs_info", frs_info);
        fun("rls_info", rls_info);
        fun("rrs_info", rrs_info);
        fun("flc_info", flc_info);
        fun("flm_info", flm_info);
        fun("frm_info", frm_info);
        fun("frc_info", frc_info);
        fun("rlc_info", rlc_info);
        fun("rlm_info", rlm_info);
        fun("rrm_info", rrm_info);
        fun("rrc_info", rrc_info);
        fun("counter", counter);
        fun("pdcinfo_avm", pdcinfo_avm);
        fun("pdcinfo_filter", pdcinfo_filter);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time_stamp", time_stamp);
        fun("fls_info", fls_info);
        fun("frs_info", frs_info);
        fun("rls_info", rls_info);
        fun("rrs_info", rrs_info);
        fun("flc_info", flc_info);
        fun("flm_info", flm_info);
        fun("frm_info", frm_info);
        fun("frc_info", frc_info);
        fun("rlc_info", rlc_info);
        fun("rlm_info", rlm_info);
        fun("rrm_info", rrm_info);
        fun("rrc_info", rrc_info);
        fun("counter", counter);
        fun("pdcinfo_avm", pdcinfo_avm);
        fun("pdcinfo_filter", pdcinfo_filter);
    }

    bool operator==(const ::hozon::sensors::UssRawDataSet& t) const
    {
        return (time_stamp == t.time_stamp) && (fls_info == t.fls_info) && (frs_info == t.frs_info) && (rls_info == t.rls_info) && (rrs_info == t.rrs_info) && (flc_info == t.flc_info) && (flm_info == t.flm_info) && (frm_info == t.frm_info) && (frc_info == t.frc_info) && (rlc_info == t.rlc_info) && (rlm_info == t.rlm_info) && (rrm_info == t.rrm_info) && (rrc_info == t.rrc_info) && (counter == t.counter) && (pdcinfo_avm == t.pdcinfo_avm) && (pdcinfo_filter == t.pdcinfo_filter);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_USSRAWDATASET_H
