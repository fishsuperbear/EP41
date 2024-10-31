/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOWSPEEDBSD_IMPL_TYPE_LOWSPEEDBSDSIGNAL_H
#define HOZON_LOWSPEEDBSD_IMPL_TYPE_LOWSPEEDBSDSIGNAL_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/lowspeedbsd/impl_type_objectinfoarray_6.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace lowspeedbsd {
struct LowSpeedBsdSignal {
    ::hozon::common::CommonHeader header;
    ::hozon::lowspeedbsd::objectInfoArray_6 object;
    ::UInt8 ViewSide;
    ::UInt8 Workst;

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
        fun(ViewSide);
        fun(Workst);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(object);
        fun(ViewSide);
        fun(Workst);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("object", object);
        fun("ViewSide", ViewSide);
        fun("Workst", Workst);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("object", object);
        fun("ViewSide", ViewSide);
        fun("Workst", Workst);
    }

    bool operator==(const ::hozon::lowspeedbsd::LowSpeedBsdSignal& t) const
    {
        return (header == t.header) && (object == t.object) && (ViewSide == t.ViewSide) && (Workst == t.Workst);
    }
};
} // namespace lowspeedbsd
} // namespace hozon


#endif // HOZON_LOWSPEEDBSD_IMPL_TYPE_LOWSPEEDBSDSIGNAL_H
