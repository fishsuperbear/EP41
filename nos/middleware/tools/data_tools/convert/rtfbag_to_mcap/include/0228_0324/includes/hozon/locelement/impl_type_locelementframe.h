/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCELEMENT_IMPL_TYPE_LOCELEMENTFRAME_H
#define HOZON_LOCELEMENT_IMPL_TYPE_LOCELEMENTFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint8.h"
#include "hozon/locelement/impl_type_locelementvector.h"

namespace hozon {
namespace locelement {
struct locElementFrame {
    ::hozon::common::CommonHeader header;
    ::UInt8 sensorStatus;
    ::hozon::locelement::locElementVector locElements;
    bool isValid;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(sensorStatus);
        fun(locElements);
        fun(isValid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(sensorStatus);
        fun(locElements);
        fun(isValid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("sensorStatus", sensorStatus);
        fun("locElements", locElements);
        fun("isValid", isValid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("sensorStatus", sensorStatus);
        fun("locElements", locElements);
        fun("isValid", isValid);
    }

    bool operator==(const ::hozon::locelement::locElementFrame& t) const
    {
        return (header == t.header) && (sensorStatus == t.sensorStatus) && (locElements == t.locElements) && (isValid == t.isValid);
    }
};
} // namespace locelement
} // namespace hozon


#endif // HOZON_LOCELEMENT_IMPL_TYPE_LOCELEMENTFRAME_H
