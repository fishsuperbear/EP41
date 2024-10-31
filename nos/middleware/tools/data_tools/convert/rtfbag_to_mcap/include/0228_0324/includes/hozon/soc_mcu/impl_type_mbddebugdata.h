/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDDEBUGDATA_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDDEBUGDATA_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_mbdposcalcdebug.h"
#include "hozon/soc_mcu/impl_type_mbdtrajcalcdebug.h"
#include "hozon/soc_mcu/impl_type_mbdadascalcdebug.h"
#include "hozon/soc_mcu/impl_type_mbdctrldecdebug.h"
#include "hozon/soc_mcu/impl_type_mbdlonctrldebug.h"
#include "hozon/soc_mcu/impl_type_mbdlatctrldebug.h"
#include "hozon/soc_mcu/impl_type_mbdctrloutputdebug.h"
#include "hozon/soc_mcu/impl_type_uint8array_20.h"

namespace hozon {
namespace soc_mcu {
struct MbdDebugData {
    ::hozon::soc_mcu::MbdPosCalcDebug pos_calc_debug;
    ::hozon::soc_mcu::MbdTrajCalcDebug traj_calc_debug;
    ::hozon::soc_mcu::MbdADASCalcDebug adas_calc_debug;
    ::hozon::soc_mcu::MbdCtrlDecDebug ctrl_dec_debug;
    ::hozon::soc_mcu::MbdLonCtrlDebug lon_ctrl_debug;
    ::hozon::soc_mcu::MbdLatCtrlDebug lat_ctrl_debug;
    ::hozon::soc_mcu::MbdCtrlOutputDebug ctrl_output_debug;
    ::hozon::soc_mcu::uint8Array_20 VersionNumber;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pos_calc_debug);
        fun(traj_calc_debug);
        fun(adas_calc_debug);
        fun(ctrl_dec_debug);
        fun(lon_ctrl_debug);
        fun(lat_ctrl_debug);
        fun(ctrl_output_debug);
        fun(VersionNumber);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pos_calc_debug);
        fun(traj_calc_debug);
        fun(adas_calc_debug);
        fun(ctrl_dec_debug);
        fun(lon_ctrl_debug);
        fun(lat_ctrl_debug);
        fun(ctrl_output_debug);
        fun(VersionNumber);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pos_calc_debug", pos_calc_debug);
        fun("traj_calc_debug", traj_calc_debug);
        fun("adas_calc_debug", adas_calc_debug);
        fun("ctrl_dec_debug", ctrl_dec_debug);
        fun("lon_ctrl_debug", lon_ctrl_debug);
        fun("lat_ctrl_debug", lat_ctrl_debug);
        fun("ctrl_output_debug", ctrl_output_debug);
        fun("VersionNumber", VersionNumber);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pos_calc_debug", pos_calc_debug);
        fun("traj_calc_debug", traj_calc_debug);
        fun("adas_calc_debug", adas_calc_debug);
        fun("ctrl_dec_debug", ctrl_dec_debug);
        fun("lon_ctrl_debug", lon_ctrl_debug);
        fun("lat_ctrl_debug", lat_ctrl_debug);
        fun("ctrl_output_debug", ctrl_output_debug);
        fun("VersionNumber", VersionNumber);
    }

    bool operator==(const ::hozon::soc_mcu::MbdDebugData& t) const
    {
        return (pos_calc_debug == t.pos_calc_debug) && (traj_calc_debug == t.traj_calc_debug) && (adas_calc_debug == t.adas_calc_debug) && (ctrl_dec_debug == t.ctrl_dec_debug) && (lon_ctrl_debug == t.lon_ctrl_debug) && (lat_ctrl_debug == t.lat_ctrl_debug) && (ctrl_output_debug == t.ctrl_output_debug) && (VersionNumber == t.VersionNumber);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDDEBUGDATA_H
