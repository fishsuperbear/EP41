/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOWSPEEDBSD_IMPL_TYPE_OBJECTINFO_H
#define HOZON_LOWSPEEDBSD_IMPL_TYPE_OBJECTINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace lowspeedbsd {
struct objectInfo {
    ::hozon::common::CommonHeader header;
    ::UInt8 objectID;
    ::UInt8 objectZone;
    ::UInt8 objectDirection;
    ::UInt8 warningLevel;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(objectID);
        fun(objectZone);
        fun(objectDirection);
        fun(warningLevel);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(objectID);
        fun(objectZone);
        fun(objectDirection);
        fun(warningLevel);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("objectID", objectID);
        fun("objectZone", objectZone);
        fun("objectDirection", objectDirection);
        fun("warningLevel", warningLevel);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("objectID", objectID);
        fun("objectZone", objectZone);
        fun("objectDirection", objectDirection);
        fun("warningLevel", warningLevel);
    }

    bool operator==(const ::hozon::lowspeedbsd::objectInfo& t) const
    {
        return (header == t.header) && (objectID == t.objectID) && (objectZone == t.objectZone) && (objectDirection == t.objectDirection) && (warningLevel == t.warningLevel);
    }
};
} // namespace lowspeedbsd
} // namespace hozon


#endif // HOZON_LOWSPEEDBSD_IMPL_TYPE_OBJECTINFO_H
