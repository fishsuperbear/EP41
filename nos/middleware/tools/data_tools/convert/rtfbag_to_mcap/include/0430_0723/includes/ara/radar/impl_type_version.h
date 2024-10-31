/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RADAR_IMPL_TYPE_VERSION_H
#define ARA_RADAR_IMPL_TYPE_VERSION_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace radar {
struct Version {
    ::UInt8 major;
    ::UInt8 minor;
    ::UInt8 patch;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(major);
        fun(minor);
        fun(patch);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(major);
        fun(minor);
        fun(patch);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("major", major);
        fun("minor", minor);
        fun("patch", patch);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("major", major);
        fun("minor", minor);
        fun("patch", patch);
    }

    bool operator==(const ::ara::radar::Version& t) const
    {
        return (major == t.major) && (minor == t.minor) && (patch == t.patch);
    }
};
} // namespace radar
} // namespace ara


#endif // ARA_RADAR_IMPL_TYPE_VERSION_H
