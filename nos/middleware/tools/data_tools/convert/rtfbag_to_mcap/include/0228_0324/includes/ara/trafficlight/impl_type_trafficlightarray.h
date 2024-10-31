/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHTARRAY_H
#define ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHTARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/trafficlight/impl_type_trafficlightheader.h"
#include "ara/trafficlight/impl_type_trafficlightvector.h"
#include "impl_type_int32_t.h"

namespace ara {
namespace trafficlight {
struct TrafficLightArray {
    ::ara::trafficlight::TrafficLightHeader header;
    ::ara::trafficlight::TrafficLightVector object_list;
    ::int32_t count;
    bool is_valid;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(object_list);
        fun(count);
        fun(is_valid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(object_list);
        fun(count);
        fun(is_valid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("object_list", object_list);
        fun("count", count);
        fun("is_valid", is_valid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("object_list", object_list);
        fun("count", count);
        fun("is_valid", is_valid);
    }

    bool operator==(const ::ara::trafficlight::TrafficLightArray& t) const
    {
        return (header == t.header) && (object_list == t.object_list) && (count == t.count) && (is_valid == t.is_valid);
    }
};
} // namespace trafficlight
} // namespace ara


#endif // ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHTARRAY_H
