/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_VISLANENEARRIGHTINDIVIDUALDATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_VISLANENEARRIGHTINDIVIDUALDATATYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace eq3 {
struct VisLaneNearRightIndividualDataType {
    double vis_lane_right_individ_a0;
    double vis_lane_right_individ_a2;
    ::uint8_t vis_lane_right_individ_color;
    double vis_lane_right_individ_a1;
    double vis_lane_right_individ_range;
    double vis_lane_right_individ_a3;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_lane_right_individ_a0);
        fun(vis_lane_right_individ_a2);
        fun(vis_lane_right_individ_color);
        fun(vis_lane_right_individ_a1);
        fun(vis_lane_right_individ_range);
        fun(vis_lane_right_individ_a3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_lane_right_individ_a0);
        fun(vis_lane_right_individ_a2);
        fun(vis_lane_right_individ_color);
        fun(vis_lane_right_individ_a1);
        fun(vis_lane_right_individ_range);
        fun(vis_lane_right_individ_a3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_lane_right_individ_a0", vis_lane_right_individ_a0);
        fun("vis_lane_right_individ_a2", vis_lane_right_individ_a2);
        fun("vis_lane_right_individ_color", vis_lane_right_individ_color);
        fun("vis_lane_right_individ_a1", vis_lane_right_individ_a1);
        fun("vis_lane_right_individ_range", vis_lane_right_individ_range);
        fun("vis_lane_right_individ_a3", vis_lane_right_individ_a3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_lane_right_individ_a0", vis_lane_right_individ_a0);
        fun("vis_lane_right_individ_a2", vis_lane_right_individ_a2);
        fun("vis_lane_right_individ_color", vis_lane_right_individ_color);
        fun("vis_lane_right_individ_a1", vis_lane_right_individ_a1);
        fun("vis_lane_right_individ_range", vis_lane_right_individ_range);
        fun("vis_lane_right_individ_a3", vis_lane_right_individ_a3);
    }

    bool operator==(const ::hozon::eq3::VisLaneNearRightIndividualDataType& t) const
    {
        return (fabs(static_cast<double>(vis_lane_right_individ_a0 - t.vis_lane_right_individ_a0)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_lane_right_individ_a2 - t.vis_lane_right_individ_a2)) < DBL_EPSILON) && (vis_lane_right_individ_color == t.vis_lane_right_individ_color) && (fabs(static_cast<double>(vis_lane_right_individ_a1 - t.vis_lane_right_individ_a1)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_lane_right_individ_range - t.vis_lane_right_individ_range)) < DBL_EPSILON) && (fabs(static_cast<double>(vis_lane_right_individ_a3 - t.vis_lane_right_individ_a3)) < DBL_EPSILON);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_VISLANENEARRIGHTINDIVIDUALDATATYPE_H
