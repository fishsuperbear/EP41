/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_ESTOP_H
#define ADSFI_IMPL_TYPE_ESTOP_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "ara/common/impl_type_commonheader.h"

namespace adsfi {
struct Estop {
    ::UInt8 isStop;
    ::String description;
    ::ara::common::CommonHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isStop);
        fun(description);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isStop);
        fun(description);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("isStop", isStop);
        fun("description", description);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("isStop", isStop);
        fun("description", description);
        fun("header", header);
    }

    bool operator==(const ::adsfi::Estop& t) const
    {
        return (isStop == t.isStop) && (description == t.description) && (header == t.header);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_ESTOP_H
