/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MBDTRAJCALCDEBUG_H
#define HOZON_SOC_MCU_IMPL_TYPE_MBDTRAJCALCDEBUG_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct MbdTrajCalcDebug {
    ::Boolean trajCalc_trajdata_replaning_flag;
    ::UInt32 trajcalc_trajdata_estop;
    ::UInt32 trajcalc_trajdata_gearcmd;
    ::UInt32 trajcalc_inputdata_valid;
    ::Double trajcalc_trajdata_timestamp;
    ::Double trajcalc_globaltime_timestamp;
    ::Boolean trajcalc_trajdata_pointtime_check;
    ::Boolean trajcalc_trajdata_timecheck;
    ::UInt32 trajcalc_enable;
    ::UInt32 trajcalc_lon_startpoint_index;
    ::Float trajcalc_lon_linear_ratio;
    ::Float trajcalc_lon_poserrcmd;
    ::Float trajcalc_lon_headingcmd;
    ::Float trajcalc_lon_velcmd;
    ::Float trajcalc_lon_acc_cmd;
    ::Float trajcalc_lon_curvcmd;
    ::UInt32 trajcalc_lonpre_startpoint_index;
    ::Float trajcalc_lonpre_linear_ratio;
    ::Float trajcalc_lonpre_poserrcmd;
    ::Float trajcalc_lonpre_headingcmd;
    ::Float trajcalc_lonpre_velcmd;
    ::Float trajcalc_lonpre_acc_cmd;
    ::Float trajcalc_lonpre_curvrmd;
    ::Double trajcalc_posedata_posex;
    ::Double trajcalc_posedata_posey;
    ::UInt32 trajcalc_lat_startpoint_index;
    ::Float trajcalc_lat_linear_ratio;
    ::Float trajcalc_lat_match_pointx;
    ::Float trajcalc_lat_match_pointy;
    ::Float trajcalc_lat_poserrcmd;
    ::Float trajcalc_lat_headingcmd;
    ::Float trajcalc_lat_velcmd;
    ::Float trajcalc_lat_acc_cmd;
    ::Float trajcalc_lat_curvcmd;
    ::Double trajcalc_posedata_preposex;
    ::Double trajcalc_posedata_preposey;
    ::UInt32 trajcalc_latpre_startpoint_index;
    ::Float trajcalc_latpre_linear_ratio;
    ::Float trajcalc_latpre_match_pointx;
    ::Float trajcalc_latpre_match_pointy;
    ::Float trajcalc_latpre_poserrcmd;
    ::Float trajcalc_latpre_headingcmd;
    ::Float trajcalc_latpre_velcmd;
    ::Float trajcalc_latpre_acc_cmd;
    ::Float trajcalc_latpre_curvcmd;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(trajCalc_trajdata_replaning_flag);
        fun(trajcalc_trajdata_estop);
        fun(trajcalc_trajdata_gearcmd);
        fun(trajcalc_inputdata_valid);
        fun(trajcalc_trajdata_timestamp);
        fun(trajcalc_globaltime_timestamp);
        fun(trajcalc_trajdata_pointtime_check);
        fun(trajcalc_trajdata_timecheck);
        fun(trajcalc_enable);
        fun(trajcalc_lon_startpoint_index);
        fun(trajcalc_lon_linear_ratio);
        fun(trajcalc_lon_poserrcmd);
        fun(trajcalc_lon_headingcmd);
        fun(trajcalc_lon_velcmd);
        fun(trajcalc_lon_acc_cmd);
        fun(trajcalc_lon_curvcmd);
        fun(trajcalc_lonpre_startpoint_index);
        fun(trajcalc_lonpre_linear_ratio);
        fun(trajcalc_lonpre_poserrcmd);
        fun(trajcalc_lonpre_headingcmd);
        fun(trajcalc_lonpre_velcmd);
        fun(trajcalc_lonpre_acc_cmd);
        fun(trajcalc_lonpre_curvrmd);
        fun(trajcalc_posedata_posex);
        fun(trajcalc_posedata_posey);
        fun(trajcalc_lat_startpoint_index);
        fun(trajcalc_lat_linear_ratio);
        fun(trajcalc_lat_match_pointx);
        fun(trajcalc_lat_match_pointy);
        fun(trajcalc_lat_poserrcmd);
        fun(trajcalc_lat_headingcmd);
        fun(trajcalc_lat_velcmd);
        fun(trajcalc_lat_acc_cmd);
        fun(trajcalc_lat_curvcmd);
        fun(trajcalc_posedata_preposex);
        fun(trajcalc_posedata_preposey);
        fun(trajcalc_latpre_startpoint_index);
        fun(trajcalc_latpre_linear_ratio);
        fun(trajcalc_latpre_match_pointx);
        fun(trajcalc_latpre_match_pointy);
        fun(trajcalc_latpre_poserrcmd);
        fun(trajcalc_latpre_headingcmd);
        fun(trajcalc_latpre_velcmd);
        fun(trajcalc_latpre_acc_cmd);
        fun(trajcalc_latpre_curvcmd);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(trajCalc_trajdata_replaning_flag);
        fun(trajcalc_trajdata_estop);
        fun(trajcalc_trajdata_gearcmd);
        fun(trajcalc_inputdata_valid);
        fun(trajcalc_trajdata_timestamp);
        fun(trajcalc_globaltime_timestamp);
        fun(trajcalc_trajdata_pointtime_check);
        fun(trajcalc_trajdata_timecheck);
        fun(trajcalc_enable);
        fun(trajcalc_lon_startpoint_index);
        fun(trajcalc_lon_linear_ratio);
        fun(trajcalc_lon_poserrcmd);
        fun(trajcalc_lon_headingcmd);
        fun(trajcalc_lon_velcmd);
        fun(trajcalc_lon_acc_cmd);
        fun(trajcalc_lon_curvcmd);
        fun(trajcalc_lonpre_startpoint_index);
        fun(trajcalc_lonpre_linear_ratio);
        fun(trajcalc_lonpre_poserrcmd);
        fun(trajcalc_lonpre_headingcmd);
        fun(trajcalc_lonpre_velcmd);
        fun(trajcalc_lonpre_acc_cmd);
        fun(trajcalc_lonpre_curvrmd);
        fun(trajcalc_posedata_posex);
        fun(trajcalc_posedata_posey);
        fun(trajcalc_lat_startpoint_index);
        fun(trajcalc_lat_linear_ratio);
        fun(trajcalc_lat_match_pointx);
        fun(trajcalc_lat_match_pointy);
        fun(trajcalc_lat_poserrcmd);
        fun(trajcalc_lat_headingcmd);
        fun(trajcalc_lat_velcmd);
        fun(trajcalc_lat_acc_cmd);
        fun(trajcalc_lat_curvcmd);
        fun(trajcalc_posedata_preposex);
        fun(trajcalc_posedata_preposey);
        fun(trajcalc_latpre_startpoint_index);
        fun(trajcalc_latpre_linear_ratio);
        fun(trajcalc_latpre_match_pointx);
        fun(trajcalc_latpre_match_pointy);
        fun(trajcalc_latpre_poserrcmd);
        fun(trajcalc_latpre_headingcmd);
        fun(trajcalc_latpre_velcmd);
        fun(trajcalc_latpre_acc_cmd);
        fun(trajcalc_latpre_curvcmd);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("trajCalc_trajdata_replaning_flag", trajCalc_trajdata_replaning_flag);
        fun("trajcalc_trajdata_estop", trajcalc_trajdata_estop);
        fun("trajcalc_trajdata_gearcmd", trajcalc_trajdata_gearcmd);
        fun("trajcalc_inputdata_valid", trajcalc_inputdata_valid);
        fun("trajcalc_trajdata_timestamp", trajcalc_trajdata_timestamp);
        fun("trajcalc_globaltime_timestamp", trajcalc_globaltime_timestamp);
        fun("trajcalc_trajdata_pointtime_check", trajcalc_trajdata_pointtime_check);
        fun("trajcalc_trajdata_timecheck", trajcalc_trajdata_timecheck);
        fun("trajcalc_enable", trajcalc_enable);
        fun("trajcalc_lon_startpoint_index", trajcalc_lon_startpoint_index);
        fun("trajcalc_lon_linear_ratio", trajcalc_lon_linear_ratio);
        fun("trajcalc_lon_poserrcmd", trajcalc_lon_poserrcmd);
        fun("trajcalc_lon_headingcmd", trajcalc_lon_headingcmd);
        fun("trajcalc_lon_velcmd", trajcalc_lon_velcmd);
        fun("trajcalc_lon_acc_cmd", trajcalc_lon_acc_cmd);
        fun("trajcalc_lon_curvcmd", trajcalc_lon_curvcmd);
        fun("trajcalc_lonpre_startpoint_index", trajcalc_lonpre_startpoint_index);
        fun("trajcalc_lonpre_linear_ratio", trajcalc_lonpre_linear_ratio);
        fun("trajcalc_lonpre_poserrcmd", trajcalc_lonpre_poserrcmd);
        fun("trajcalc_lonpre_headingcmd", trajcalc_lonpre_headingcmd);
        fun("trajcalc_lonpre_velcmd", trajcalc_lonpre_velcmd);
        fun("trajcalc_lonpre_acc_cmd", trajcalc_lonpre_acc_cmd);
        fun("trajcalc_lonpre_curvrmd", trajcalc_lonpre_curvrmd);
        fun("trajcalc_posedata_posex", trajcalc_posedata_posex);
        fun("trajcalc_posedata_posey", trajcalc_posedata_posey);
        fun("trajcalc_lat_startpoint_index", trajcalc_lat_startpoint_index);
        fun("trajcalc_lat_linear_ratio", trajcalc_lat_linear_ratio);
        fun("trajcalc_lat_match_pointx", trajcalc_lat_match_pointx);
        fun("trajcalc_lat_match_pointy", trajcalc_lat_match_pointy);
        fun("trajcalc_lat_poserrcmd", trajcalc_lat_poserrcmd);
        fun("trajcalc_lat_headingcmd", trajcalc_lat_headingcmd);
        fun("trajcalc_lat_velcmd", trajcalc_lat_velcmd);
        fun("trajcalc_lat_acc_cmd", trajcalc_lat_acc_cmd);
        fun("trajcalc_lat_curvcmd", trajcalc_lat_curvcmd);
        fun("trajcalc_posedata_preposex", trajcalc_posedata_preposex);
        fun("trajcalc_posedata_preposey", trajcalc_posedata_preposey);
        fun("trajcalc_latpre_startpoint_index", trajcalc_latpre_startpoint_index);
        fun("trajcalc_latpre_linear_ratio", trajcalc_latpre_linear_ratio);
        fun("trajcalc_latpre_match_pointx", trajcalc_latpre_match_pointx);
        fun("trajcalc_latpre_match_pointy", trajcalc_latpre_match_pointy);
        fun("trajcalc_latpre_poserrcmd", trajcalc_latpre_poserrcmd);
        fun("trajcalc_latpre_headingcmd", trajcalc_latpre_headingcmd);
        fun("trajcalc_latpre_velcmd", trajcalc_latpre_velcmd);
        fun("trajcalc_latpre_acc_cmd", trajcalc_latpre_acc_cmd);
        fun("trajcalc_latpre_curvcmd", trajcalc_latpre_curvcmd);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("trajCalc_trajdata_replaning_flag", trajCalc_trajdata_replaning_flag);
        fun("trajcalc_trajdata_estop", trajcalc_trajdata_estop);
        fun("trajcalc_trajdata_gearcmd", trajcalc_trajdata_gearcmd);
        fun("trajcalc_inputdata_valid", trajcalc_inputdata_valid);
        fun("trajcalc_trajdata_timestamp", trajcalc_trajdata_timestamp);
        fun("trajcalc_globaltime_timestamp", trajcalc_globaltime_timestamp);
        fun("trajcalc_trajdata_pointtime_check", trajcalc_trajdata_pointtime_check);
        fun("trajcalc_trajdata_timecheck", trajcalc_trajdata_timecheck);
        fun("trajcalc_enable", trajcalc_enable);
        fun("trajcalc_lon_startpoint_index", trajcalc_lon_startpoint_index);
        fun("trajcalc_lon_linear_ratio", trajcalc_lon_linear_ratio);
        fun("trajcalc_lon_poserrcmd", trajcalc_lon_poserrcmd);
        fun("trajcalc_lon_headingcmd", trajcalc_lon_headingcmd);
        fun("trajcalc_lon_velcmd", trajcalc_lon_velcmd);
        fun("trajcalc_lon_acc_cmd", trajcalc_lon_acc_cmd);
        fun("trajcalc_lon_curvcmd", trajcalc_lon_curvcmd);
        fun("trajcalc_lonpre_startpoint_index", trajcalc_lonpre_startpoint_index);
        fun("trajcalc_lonpre_linear_ratio", trajcalc_lonpre_linear_ratio);
        fun("trajcalc_lonpre_poserrcmd", trajcalc_lonpre_poserrcmd);
        fun("trajcalc_lonpre_headingcmd", trajcalc_lonpre_headingcmd);
        fun("trajcalc_lonpre_velcmd", trajcalc_lonpre_velcmd);
        fun("trajcalc_lonpre_acc_cmd", trajcalc_lonpre_acc_cmd);
        fun("trajcalc_lonpre_curvrmd", trajcalc_lonpre_curvrmd);
        fun("trajcalc_posedata_posex", trajcalc_posedata_posex);
        fun("trajcalc_posedata_posey", trajcalc_posedata_posey);
        fun("trajcalc_lat_startpoint_index", trajcalc_lat_startpoint_index);
        fun("trajcalc_lat_linear_ratio", trajcalc_lat_linear_ratio);
        fun("trajcalc_lat_match_pointx", trajcalc_lat_match_pointx);
        fun("trajcalc_lat_match_pointy", trajcalc_lat_match_pointy);
        fun("trajcalc_lat_poserrcmd", trajcalc_lat_poserrcmd);
        fun("trajcalc_lat_headingcmd", trajcalc_lat_headingcmd);
        fun("trajcalc_lat_velcmd", trajcalc_lat_velcmd);
        fun("trajcalc_lat_acc_cmd", trajcalc_lat_acc_cmd);
        fun("trajcalc_lat_curvcmd", trajcalc_lat_curvcmd);
        fun("trajcalc_posedata_preposex", trajcalc_posedata_preposex);
        fun("trajcalc_posedata_preposey", trajcalc_posedata_preposey);
        fun("trajcalc_latpre_startpoint_index", trajcalc_latpre_startpoint_index);
        fun("trajcalc_latpre_linear_ratio", trajcalc_latpre_linear_ratio);
        fun("trajcalc_latpre_match_pointx", trajcalc_latpre_match_pointx);
        fun("trajcalc_latpre_match_pointy", trajcalc_latpre_match_pointy);
        fun("trajcalc_latpre_poserrcmd", trajcalc_latpre_poserrcmd);
        fun("trajcalc_latpre_headingcmd", trajcalc_latpre_headingcmd);
        fun("trajcalc_latpre_velcmd", trajcalc_latpre_velcmd);
        fun("trajcalc_latpre_acc_cmd", trajcalc_latpre_acc_cmd);
        fun("trajcalc_latpre_curvcmd", trajcalc_latpre_curvcmd);
    }

    bool operator==(const ::hozon::soc_mcu::MbdTrajCalcDebug& t) const
    {
        return (trajCalc_trajdata_replaning_flag == t.trajCalc_trajdata_replaning_flag) && (trajcalc_trajdata_estop == t.trajcalc_trajdata_estop) && (trajcalc_trajdata_gearcmd == t.trajcalc_trajdata_gearcmd) && (trajcalc_inputdata_valid == t.trajcalc_inputdata_valid) && (fabs(static_cast<double>(trajcalc_trajdata_timestamp - t.trajcalc_trajdata_timestamp)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_globaltime_timestamp - t.trajcalc_globaltime_timestamp)) < DBL_EPSILON) && (trajcalc_trajdata_pointtime_check == t.trajcalc_trajdata_pointtime_check) && (trajcalc_trajdata_timecheck == t.trajcalc_trajdata_timecheck) && (trajcalc_enable == t.trajcalc_enable) && (trajcalc_lon_startpoint_index == t.trajcalc_lon_startpoint_index) && (fabs(static_cast<double>(trajcalc_lon_linear_ratio - t.trajcalc_lon_linear_ratio)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lon_poserrcmd - t.trajcalc_lon_poserrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lon_headingcmd - t.trajcalc_lon_headingcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lon_velcmd - t.trajcalc_lon_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lon_acc_cmd - t.trajcalc_lon_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lon_curvcmd - t.trajcalc_lon_curvcmd)) < DBL_EPSILON) && (trajcalc_lonpre_startpoint_index == t.trajcalc_lonpre_startpoint_index) && (fabs(static_cast<double>(trajcalc_lonpre_linear_ratio - t.trajcalc_lonpre_linear_ratio)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lonpre_poserrcmd - t.trajcalc_lonpre_poserrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lonpre_headingcmd - t.trajcalc_lonpre_headingcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lonpre_velcmd - t.trajcalc_lonpre_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lonpre_acc_cmd - t.trajcalc_lonpre_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lonpre_curvrmd - t.trajcalc_lonpre_curvrmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_posedata_posex - t.trajcalc_posedata_posex)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_posedata_posey - t.trajcalc_posedata_posey)) < DBL_EPSILON) && (trajcalc_lat_startpoint_index == t.trajcalc_lat_startpoint_index) && (fabs(static_cast<double>(trajcalc_lat_linear_ratio - t.trajcalc_lat_linear_ratio)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_match_pointx - t.trajcalc_lat_match_pointx)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_match_pointy - t.trajcalc_lat_match_pointy)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_poserrcmd - t.trajcalc_lat_poserrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_headingcmd - t.trajcalc_lat_headingcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_velcmd - t.trajcalc_lat_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_acc_cmd - t.trajcalc_lat_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_lat_curvcmd - t.trajcalc_lat_curvcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_posedata_preposex - t.trajcalc_posedata_preposex)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_posedata_preposey - t.trajcalc_posedata_preposey)) < DBL_EPSILON) && (trajcalc_latpre_startpoint_index == t.trajcalc_latpre_startpoint_index) && (fabs(static_cast<double>(trajcalc_latpre_linear_ratio - t.trajcalc_latpre_linear_ratio)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_match_pointx - t.trajcalc_latpre_match_pointx)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_match_pointy - t.trajcalc_latpre_match_pointy)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_poserrcmd - t.trajcalc_latpre_poserrcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_headingcmd - t.trajcalc_latpre_headingcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_velcmd - t.trajcalc_latpre_velcmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_acc_cmd - t.trajcalc_latpre_acc_cmd)) < DBL_EPSILON) && (fabs(static_cast<double>(trajcalc_latpre_curvcmd - t.trajcalc_latpre_curvcmd)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MBDTRAJCALCDEBUG_H
