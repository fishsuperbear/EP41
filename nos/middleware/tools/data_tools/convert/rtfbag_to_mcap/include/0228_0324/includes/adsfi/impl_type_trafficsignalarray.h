/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_TRAFFICSIGNALARRAY_H
#define ADSFI_IMPL_TYPE_TRAFFICSIGNALARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "adsfi/impl_type_trafficsignalvector.h"

namespace adsfi {
struct TrafficSignalArray {
    ::ara::common::CommonHeader header;
    ::adsfi::TrafficSignalVector tlObject;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(tlObject);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(tlObject);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("tlObject", tlObject);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("tlObject", tlObject);
    }

    bool operator==(const ::adsfi::TrafficSignalArray& t) const
    {
        return (header == t.header) && (tlObject == t.tlObject);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_TRAFFICSIGNALARRAY_H
