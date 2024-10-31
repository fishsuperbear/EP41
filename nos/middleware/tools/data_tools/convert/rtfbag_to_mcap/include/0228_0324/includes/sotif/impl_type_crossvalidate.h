/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef SOTIF_IMPL_TYPE_CROSSVALIDATE_H
#define SOTIF_IMPL_TYPE_CROSSVALIDATE_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commontime.h"
#include "impl_type_uint32.h"
#include "impl_type_stringvector.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_floatarray.h"

namespace sotif {
struct CrossValidate {
    ::ara::common::CommonTime timeStampFirst;
    ::UInt32 eventId;
    ::StringVector sensorIdList;
    ::UInt8 eventLevel;
    ::UInt32 type;
    ::String actionName;
    ::FloatArray para;
    ::ara::common::CommonTime timeStampLast;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(timeStampFirst);
        fun(eventId);
        fun(sensorIdList);
        fun(eventLevel);
        fun(type);
        fun(actionName);
        fun(para);
        fun(timeStampLast);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(timeStampFirst);
        fun(eventId);
        fun(sensorIdList);
        fun(eventLevel);
        fun(type);
        fun(actionName);
        fun(para);
        fun(timeStampLast);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("timeStampFirst", timeStampFirst);
        fun("eventId", eventId);
        fun("sensorIdList", sensorIdList);
        fun("eventLevel", eventLevel);
        fun("type", type);
        fun("actionName", actionName);
        fun("para", para);
        fun("timeStampLast", timeStampLast);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("timeStampFirst", timeStampFirst);
        fun("eventId", eventId);
        fun("sensorIdList", sensorIdList);
        fun("eventLevel", eventLevel);
        fun("type", type);
        fun("actionName", actionName);
        fun("para", para);
        fun("timeStampLast", timeStampLast);
    }

    bool operator==(const ::sotif::CrossValidate& t) const
    {
        return (timeStampFirst == t.timeStampFirst) && (eventId == t.eventId) && (sensorIdList == t.sensorIdList) && (eventLevel == t.eventLevel) && (type == t.type) && (actionName == t.actionName) && (para == t.para) && (timeStampLast == t.timeStampLast);
    }
};
} // namespace sotif


#endif // SOTIF_IMPL_TYPE_CROSSVALIDATE_H
