/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_FEATURE_H
#define HOZON_OBJECT_IMPL_TYPE_FEATURE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32vector.h"
#include "impl_type_doublevector.h"

namespace hozon {
namespace object {
struct Feature {
    ::Uint32Vector shape;
    ::DoubleVector value;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(shape);
        fun(value);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(shape);
        fun(value);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("shape", shape);
        fun("value", value);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("shape", shape);
        fun("value", value);
    }

    bool operator==(const ::hozon::object::Feature& t) const
    {
        return (shape == t.shape) && (value == t.value);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_FEATURE_H
