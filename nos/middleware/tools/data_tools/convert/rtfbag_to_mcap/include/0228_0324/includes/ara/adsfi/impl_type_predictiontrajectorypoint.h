/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_PREDICTIONTRAJECTORYPOINT_H
#define ARA_ADSFI_IMPL_TYPE_PREDICTIONTRAJECTORYPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_point.h"
#include "ara/adsfi/impl_type_time.h"

namespace ara {
namespace adsfi {
struct PredictionTrajectoryPoint {
    ::Point point;
    ::ara::adsfi::Time timestamp;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(point);
        fun(timestamp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(point);
        fun(timestamp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("point", point);
        fun("timestamp", timestamp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("point", point);
        fun("timestamp", timestamp);
    }

    bool operator==(const ::ara::adsfi::PredictionTrajectoryPoint& t) const
    {
        return (point == t.point) && (timestamp == t.timestamp);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_PREDICTIONTRAJECTORYPOINT_H
