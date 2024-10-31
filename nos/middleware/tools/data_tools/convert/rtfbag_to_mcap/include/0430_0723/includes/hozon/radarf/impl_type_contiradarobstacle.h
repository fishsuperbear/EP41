/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_RADARF_IMPL_TYPE_CONTIRADAROBSTACLE_H
#define HOZON_RADARF_IMPL_TYPE_CONTIRADAROBSTACLE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace radarf {
struct ContiRadarObstacle {
    ::uint8_t id;
    ::uint8_t motionpattern;
    double obs_prob;
    ::uint8_t measflag;
    ::uint8_t obstype;
    double longitude_dist;
    double lateral_dist;
    double longitude_v;
    double lateral_v;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(motionpattern);
        fun(obs_prob);
        fun(measflag);
        fun(obstype);
        fun(longitude_dist);
        fun(lateral_dist);
        fun(longitude_v);
        fun(lateral_v);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(motionpattern);
        fun(obs_prob);
        fun(measflag);
        fun(obstype);
        fun(longitude_dist);
        fun(lateral_dist);
        fun(longitude_v);
        fun(lateral_v);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("motionpattern", motionpattern);
        fun("obs_prob", obs_prob);
        fun("measflag", measflag);
        fun("obstype", obstype);
        fun("longitude_dist", longitude_dist);
        fun("lateral_dist", lateral_dist);
        fun("longitude_v", longitude_v);
        fun("lateral_v", lateral_v);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("motionpattern", motionpattern);
        fun("obs_prob", obs_prob);
        fun("measflag", measflag);
        fun("obstype", obstype);
        fun("longitude_dist", longitude_dist);
        fun("lateral_dist", lateral_dist);
        fun("longitude_v", longitude_v);
        fun("lateral_v", lateral_v);
    }

    bool operator==(const ::hozon::radarf::ContiRadarObstacle& t) const
    {
        return (id == t.id) && (motionpattern == t.motionpattern) && (fabs(static_cast<double>(obs_prob - t.obs_prob)) < DBL_EPSILON) && (measflag == t.measflag) && (obstype == t.obstype) && (fabs(static_cast<double>(longitude_dist - t.longitude_dist)) < DBL_EPSILON) && (fabs(static_cast<double>(lateral_dist - t.lateral_dist)) < DBL_EPSILON) && (fabs(static_cast<double>(longitude_v - t.longitude_v)) < DBL_EPSILON) && (fabs(static_cast<double>(lateral_v - t.lateral_v)) < DBL_EPSILON);
    }
};
} // namespace radarf
} // namespace hozon


#endif // HOZON_RADARF_IMPL_TYPE_CONTIRADAROBSTACLE_H
