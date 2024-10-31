/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLCFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLCFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_double.h"

namespace ara {
namespace vehicle {
struct FLCFr01Info {
    ::UInt8 flc_ldwlka_mode;
    ::UInt8 flc_ldwlka_warnsetsta;
    ::UInt8 flc_adas_sndctrl_lka_onoff_fbk;
    ::UInt8 flc_lka_handsoffwarning;
    ::UInt8 flc_ldwlka_recog_sysstate;
    ::UInt8 flc_ldwlka_symbol;
    ::UInt8 flc_ldwlka_rhwarning;
    ::UInt8 flc_ldwlka_lhwarning;
    ::UInt8 flc_lka_handsoffsndwaring;
    ::UInt8 flc_tja_sysstate;
    ::UInt8 flc_lka_toifault;
    ::UInt8 flc_lka_toiact;
    ::Double flc_lka_strtoqreq;
    ::UInt8 flc_status;
    ::UInt8 flc_tja_syswarning;
    ::UInt8 flc_fr01_msgcounter;
    ::UInt8 flc_fr01_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(flc_ldwlka_mode);
        fun(flc_ldwlka_warnsetsta);
        fun(flc_adas_sndctrl_lka_onoff_fbk);
        fun(flc_lka_handsoffwarning);
        fun(flc_ldwlka_recog_sysstate);
        fun(flc_ldwlka_symbol);
        fun(flc_ldwlka_rhwarning);
        fun(flc_ldwlka_lhwarning);
        fun(flc_lka_handsoffsndwaring);
        fun(flc_tja_sysstate);
        fun(flc_lka_toifault);
        fun(flc_lka_toiact);
        fun(flc_lka_strtoqreq);
        fun(flc_status);
        fun(flc_tja_syswarning);
        fun(flc_fr01_msgcounter);
        fun(flc_fr01_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(flc_ldwlka_mode);
        fun(flc_ldwlka_warnsetsta);
        fun(flc_adas_sndctrl_lka_onoff_fbk);
        fun(flc_lka_handsoffwarning);
        fun(flc_ldwlka_recog_sysstate);
        fun(flc_ldwlka_symbol);
        fun(flc_ldwlka_rhwarning);
        fun(flc_ldwlka_lhwarning);
        fun(flc_lka_handsoffsndwaring);
        fun(flc_tja_sysstate);
        fun(flc_lka_toifault);
        fun(flc_lka_toiact);
        fun(flc_lka_strtoqreq);
        fun(flc_status);
        fun(flc_tja_syswarning);
        fun(flc_fr01_msgcounter);
        fun(flc_fr01_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("flc_ldwlka_mode", flc_ldwlka_mode);
        fun("flc_ldwlka_warnsetsta", flc_ldwlka_warnsetsta);
        fun("flc_adas_sndctrl_lka_onoff_fbk", flc_adas_sndctrl_lka_onoff_fbk);
        fun("flc_lka_handsoffwarning", flc_lka_handsoffwarning);
        fun("flc_ldwlka_recog_sysstate", flc_ldwlka_recog_sysstate);
        fun("flc_ldwlka_symbol", flc_ldwlka_symbol);
        fun("flc_ldwlka_rhwarning", flc_ldwlka_rhwarning);
        fun("flc_ldwlka_lhwarning", flc_ldwlka_lhwarning);
        fun("flc_lka_handsoffsndwaring", flc_lka_handsoffsndwaring);
        fun("flc_tja_sysstate", flc_tja_sysstate);
        fun("flc_lka_toifault", flc_lka_toifault);
        fun("flc_lka_toiact", flc_lka_toiact);
        fun("flc_lka_strtoqreq", flc_lka_strtoqreq);
        fun("flc_status", flc_status);
        fun("flc_tja_syswarning", flc_tja_syswarning);
        fun("flc_fr01_msgcounter", flc_fr01_msgcounter);
        fun("flc_fr01_checksum", flc_fr01_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("flc_ldwlka_mode", flc_ldwlka_mode);
        fun("flc_ldwlka_warnsetsta", flc_ldwlka_warnsetsta);
        fun("flc_adas_sndctrl_lka_onoff_fbk", flc_adas_sndctrl_lka_onoff_fbk);
        fun("flc_lka_handsoffwarning", flc_lka_handsoffwarning);
        fun("flc_ldwlka_recog_sysstate", flc_ldwlka_recog_sysstate);
        fun("flc_ldwlka_symbol", flc_ldwlka_symbol);
        fun("flc_ldwlka_rhwarning", flc_ldwlka_rhwarning);
        fun("flc_ldwlka_lhwarning", flc_ldwlka_lhwarning);
        fun("flc_lka_handsoffsndwaring", flc_lka_handsoffsndwaring);
        fun("flc_tja_sysstate", flc_tja_sysstate);
        fun("flc_lka_toifault", flc_lka_toifault);
        fun("flc_lka_toiact", flc_lka_toiact);
        fun("flc_lka_strtoqreq", flc_lka_strtoqreq);
        fun("flc_status", flc_status);
        fun("flc_tja_syswarning", flc_tja_syswarning);
        fun("flc_fr01_msgcounter", flc_fr01_msgcounter);
        fun("flc_fr01_checksum", flc_fr01_checksum);
    }

    bool operator==(const ::ara::vehicle::FLCFr01Info& t) const
    {
        return (flc_ldwlka_mode == t.flc_ldwlka_mode) && (flc_ldwlka_warnsetsta == t.flc_ldwlka_warnsetsta) && (flc_adas_sndctrl_lka_onoff_fbk == t.flc_adas_sndctrl_lka_onoff_fbk) && (flc_lka_handsoffwarning == t.flc_lka_handsoffwarning) && (flc_ldwlka_recog_sysstate == t.flc_ldwlka_recog_sysstate) && (flc_ldwlka_symbol == t.flc_ldwlka_symbol) && (flc_ldwlka_rhwarning == t.flc_ldwlka_rhwarning) && (flc_ldwlka_lhwarning == t.flc_ldwlka_lhwarning) && (flc_lka_handsoffsndwaring == t.flc_lka_handsoffsndwaring) && (flc_tja_sysstate == t.flc_tja_sysstate) && (flc_lka_toifault == t.flc_lka_toifault) && (flc_lka_toiact == t.flc_lka_toiact) && (fabs(static_cast<double>(flc_lka_strtoqreq - t.flc_lka_strtoqreq)) < DBL_EPSILON) && (flc_status == t.flc_status) && (flc_tja_syswarning == t.flc_tja_syswarning) && (flc_fr01_msgcounter == t.flc_fr01_msgcounter) && (flc_fr01_checksum == t.flc_fr01_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLCFR01INFO_H
