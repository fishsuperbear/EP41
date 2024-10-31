/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef SOTIF_IMPL_TYPE_DECISIONSAFETY_H
#define SOTIF_IMPL_TYPE_DECISIONSAFETY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_floatarray.h"

namespace sotif {
struct DecisionSafety {
    ::ara::common::CommonHeader header;
    ::UInt32 eventID;
    ::UInt8 eventLevel;
    ::UInt32 type;
    ::String actionName;
    ::FloatArray para;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(eventID);
        fun(eventLevel);
        fun(type);
        fun(actionName);
        fun(para);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(eventID);
        fun(eventLevel);
        fun(type);
        fun(actionName);
        fun(para);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("eventID", eventID);
        fun("eventLevel", eventLevel);
        fun("type", type);
        fun("actionName", actionName);
        fun("para", para);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("eventID", eventID);
        fun("eventLevel", eventLevel);
        fun("type", type);
        fun("actionName", actionName);
        fun("para", para);
    }

    bool operator==(const ::sotif::DecisionSafety& t) const
    {
        return (header == t.header) && (eventID == t.eventID) && (eventLevel == t.eventLevel) && (type == t.type) && (actionName == t.actionName) && (para == t.para);
    }
};
} // namespace sotif


#endif // SOTIF_IMPL_TYPE_DECISIONSAFETY_H
