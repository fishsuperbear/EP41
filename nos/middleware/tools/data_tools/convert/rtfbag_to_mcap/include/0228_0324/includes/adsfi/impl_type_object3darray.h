/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_OBJECT3DARRAY_H
#define ADSFI_IMPL_TYPE_OBJECT3DARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "adsfi/impl_type_objec3dvector.h"

namespace adsfi {
struct Object3dArray {
    ::ara::common::CommonHeader header;
    ::adsfi::Objec3dVector object;

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

    bool operator==(const ::adsfi::Object3dArray& t) const
    {
        return (header == t.header) && (object == t.object);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_OBJECT3DARRAY_H
