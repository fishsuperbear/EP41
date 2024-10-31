/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_TRAJECTORYINPREDICTION_H
#define ARA_ADSFI_IMPL_TYPE_TRAJECTORYINPREDICTION_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "ara/adsfi/impl_type_trajectorypointvector.h"

namespace ara {
namespace adsfi {
struct TrajectoryInPrediction {
    ::Double probability;
    ::ara::adsfi::TrajectoryPointVector trajectoryPoints;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(probability);
        fun(trajectoryPoints);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(probability);
        fun(trajectoryPoints);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("probability", probability);
        fun("trajectoryPoints", trajectoryPoints);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("probability", probability);
        fun("trajectoryPoints", trajectoryPoints);
    }

    bool operator==(const ::ara::adsfi::TrajectoryInPrediction& t) const
    {
        return (fabs(static_cast<double>(probability - t.probability)) < DBL_EPSILON) && (trajectoryPoints == t.trajectoryPoints);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_TRAJECTORYINPREDICTION_H
