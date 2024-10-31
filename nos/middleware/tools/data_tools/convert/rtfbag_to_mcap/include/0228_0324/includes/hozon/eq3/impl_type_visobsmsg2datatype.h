/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_VISOBSMSG2DATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_VISOBSMSG2DATATYPE_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace eq3 {
struct VisObsMsg2DataType {
    double vis_obs_long_accel_01;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_obs_long_accel_01);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_obs_long_accel_01);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_obs_long_accel_01", vis_obs_long_accel_01);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_obs_long_accel_01", vis_obs_long_accel_01);
    }

    bool operator==(const ::hozon::eq3::VisObsMsg2DataType& t) const
    {
        return (fabs(static_cast<double>(vis_obs_long_accel_01 - t.vis_obs_long_accel_01)) < DBL_EPSILON);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_VISOBSMSG2DATATYPE_H
