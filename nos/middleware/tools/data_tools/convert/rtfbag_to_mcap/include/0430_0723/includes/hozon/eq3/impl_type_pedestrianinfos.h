/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_PEDESTRIANINFOS_H
#define HOZON_EQ3_IMPL_TYPE_PEDESTRIANINFOS_H
#include <cfloat>
#include <cmath>
#include "hozon/eq3/impl_type_pedestriandataarray.h"

namespace hozon {
namespace eq3 {
struct PedestrianInfos {
    ::hozon::eq3::PedestrianDataArray PedestrianDataArray;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(PedestrianDataArray);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(PedestrianDataArray);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("PedestrianDataArray", PedestrianDataArray);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("PedestrianDataArray", PedestrianDataArray);
    }

    bool operator==(const ::hozon::eq3::PedestrianInfos& t) const
    {
        return (PedestrianDataArray == t.PedestrianDataArray);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_PEDESTRIANINFOS_H
