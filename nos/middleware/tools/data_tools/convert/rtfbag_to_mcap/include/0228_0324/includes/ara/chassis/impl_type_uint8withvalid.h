/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_UINT8WITHVALID_H
#define ARA_CHASSIS_IMPL_TYPE_UINT8WITHVALID_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace chassis {
struct Uint8WithValid {
    ::UInt8 value;
    ::UInt8 confidence;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(value);
        fun(confidence);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(value);
        fun(confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("value", value);
        fun("confidence", confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("value", value);
        fun("confidence", confidence);
    }

    bool operator==(const ::ara::chassis::Uint8WithValid& t) const
    {
        return (value == t.value) && (confidence == t.confidence);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_UINT8WITHVALID_H