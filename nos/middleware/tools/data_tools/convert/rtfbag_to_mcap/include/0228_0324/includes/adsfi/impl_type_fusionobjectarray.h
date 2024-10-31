/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_FUSIONOBJECTARRAY_H
#define ADSFI_IMPL_TYPE_FUSIONOBJECTARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "adsfi/impl_type_fusionobjectvector.h"

namespace adsfi {
struct FusionObjectArray {
    ::ara::common::CommonHeader header;
    ::adsfi::FusionObjectVector object;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(object);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(object);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("object", object);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("object", object);
    }

    bool operator==(const ::adsfi::FusionObjectArray& t) const
    {
        return (header == t.header) && (object == t.object);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_FUSIONOBJECTARRAY_H
