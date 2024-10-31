/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOFAILED_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafLidarPercpAlgoFailed {
    ::UInt8 algo_configload_error;
    ::UInt8 algo_pointcloud_preproc_error;
    ::UInt8 algo_target_detn_error;
    ::UInt8 algo_target_track_error;
    ::UInt8 alog_grd_detn_error;
    ::UInt8 algo_lane_detn_error;
    ::UInt8 algo_freespace_detn_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(algo_configload_error);
        fun(algo_pointcloud_preproc_error);
        fun(algo_target_detn_error);
        fun(algo_target_track_error);
        fun(alog_grd_detn_error);
        fun(algo_lane_detn_error);
        fun(algo_freespace_detn_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(algo_configload_error);
        fun(algo_pointcloud_preproc_error);
        fun(algo_target_detn_error);
        fun(algo_target_track_error);
        fun(alog_grd_detn_error);
        fun(algo_lane_detn_error);
        fun(algo_freespace_detn_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("algo_configload_error", algo_configload_error);
        fun("algo_pointcloud_preproc_error", algo_pointcloud_preproc_error);
        fun("algo_target_detn_error", algo_target_detn_error);
        fun("algo_target_track_error", algo_target_track_error);
        fun("alog_grd_detn_error", alog_grd_detn_error);
        fun("algo_lane_detn_error", algo_lane_detn_error);
        fun("algo_freespace_detn_error", algo_freespace_detn_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("algo_configload_error", algo_configload_error);
        fun("algo_pointcloud_preproc_error", algo_pointcloud_preproc_error);
        fun("algo_target_detn_error", algo_target_detn_error);
        fun("algo_target_track_error", algo_target_track_error);
        fun("alog_grd_detn_error", alog_grd_detn_error);
        fun("algo_lane_detn_error", algo_lane_detn_error);
        fun("algo_freespace_detn_error", algo_freespace_detn_error);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafLidarPercpAlgoFailed& t) const
    {
        return (algo_configload_error == t.algo_configload_error) && (algo_pointcloud_preproc_error == t.algo_pointcloud_preproc_error) && (algo_target_detn_error == t.algo_target_detn_error) && (algo_target_track_error == t.algo_target_track_error) && (alog_grd_detn_error == t.alog_grd_detn_error) && (algo_lane_detn_error == t.algo_lane_detn_error) && (algo_freespace_detn_error == t.algo_freespace_detn_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOFAILED_H
