/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_OBJECTARRAY_H
#define ARA_ADSFI_IMPL_TYPE_OBJECTARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_uint8.h"
#include "ara/adsfi/impl_type_objectvector.h"

namespace ara {
namespace adsfi {
struct ObjectArray {
    ::ara::common::CommonHeader header;
    ::UInt8 package_type;
    ::ara::adsfi::ObjectVector object_list;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(package_type);
        fun(object_list);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(package_type);
        fun(object_list);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("package_type", package_type);
        fun("object_list", object_list);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("package_type", package_type);
        fun("object_list", object_list);
    }

    bool operator==(const ::ara::adsfi::ObjectArray& t) const
    {
        return (header == t.header) && (package_type == t.package_type) && (object_list == t.object_list);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_OBJECTARRAY_H
