/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_MODULEINFO_H
#define ARA_CAMERA_IMPL_TYPE_MODULEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace ara {
namespace camera {
struct ModuleInfo {
    ::String name;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(name);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(name);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("name", name);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("name", name);
    }

    bool operator==(const ::ara::camera::ModuleInfo& t) const
    {
        return (name == t.name);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_MODULEINFO_H
