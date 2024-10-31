/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_VISOBSMSG3DATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_VISOBSMSG3DATATYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"

namespace hozon {
namespace eq3 {
struct VisObsMsg3DataType {
    float vis_obs_width_01;
    ::Double vis_obs_long_pos_01;
    ::Double vis_obs_long_vel_01;
    ::Double vis_obs_lat_pos_01;
    ::Double vis_obs_lat_vel_01;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_obs_width_01);
        fun(vis_obs_long_pos_01);
        fun(vis_obs_long_vel_01);
        fun(vis_obs_lat_pos_01);
        fun(vis_obs_lat_vel_01);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_obs_width_01);
        fun(vis_obs_long_pos_01);
        fun(vis_obs_long_vel_01);
        fun(vis_obs_lat_pos_01);
        fun(vis_obs_lat_vel_01);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_obs_width_01", vis_obs_width_01);
        fun("vis_obs_long_pos_01", vis_obs_long_pos_01);
        fun("vis_obs_long_vel_01", vis_obs_long_vel_01);
        fun("vis_obs_lat_pos_01", vis_obs_lat_pos_01);
        fun("vis_obs_lat_vel_01", vis_obs_lat_vel_01);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_obs_width_01", vis_obs_width_01);
        fun("vis_obs_long_pos_01", vis_obs_long_pos_01);
        fun("vis_obs_long_vel_01", vis_obs_long_vel_01);
        fun("vis_obs_lat_pos_01", vis_obs_lat_pos_01);
        fun("vis_obs_lat_vel_01", vis_obs_lat_vel_01);
    }

    bool operator==(const ::hozon::eq3::VisObsMsg3DataType& t) const
    {
        return (fabs(static_cast<double>(vis_obs_width_01 - t.vis_obs_width_01)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_obs_long_pos_01 - t.vis_obs_long_pos_01)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_obs_long_vel_01 - t.vis_obs_long_vel_01)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_obs_lat_pos_01 - t.vis_obs_lat_pos_01)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_obs_lat_vel_01 - t.vis_obs_lat_vel_01)) < DBL_EPSILON);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_VISOBSMSG3DATATYPE_H
