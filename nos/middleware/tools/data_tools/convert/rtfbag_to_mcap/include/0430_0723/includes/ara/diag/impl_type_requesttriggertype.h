/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_IMPL_TYPE_REQUESTTRIGGERTYPE_H
#define ARA_DIAG_IMPL_TYPE_REQUESTTRIGGERTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint64.h"
#include "impl_type_uint8.h"

namespace ara {
namespace diag {
struct RequestTriggerType {
    ::String instanceSpecifier;
    ::UInt64 serailNumber;
    ::UInt8 eventType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(instanceSpecifier);
        fun(serailNumber);
        fun(eventType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(instanceSpecifier);
        fun(serailNumber);
        fun(eventType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("instanceSpecifier", instanceSpecifier);
        fun("serailNumber", serailNumber);
        fun("eventType", eventType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("instanceSpecifier", instanceSpecifier);
        fun("serailNumber", serailNumber);
        fun("eventType", eventType);
    }

    bool operator==(const ::ara::diag::RequestTriggerType& t) const
    {
        return (instanceSpecifier == t.instanceSpecifier) && (serailNumber == t.serailNumber) && (eventType == t.eventType);
    }
};
} // namespace diag
} // namespace ara


#endif // ARA_DIAG_IMPL_TYPE_REQUESTTRIGGERTYPE_H
