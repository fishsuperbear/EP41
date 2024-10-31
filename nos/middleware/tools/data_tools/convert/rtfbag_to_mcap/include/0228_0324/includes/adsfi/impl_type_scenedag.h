/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_SCENEDAG_H
#define ADSFI_IMPL_TYPE_SCENEDAG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "adsfi/impl_type_taskvector.h"
#include "impl_type_string.h"

namespace adsfi {
struct SceneDag {
    ::UInt32 id;
    ::adsfi::TaskVector tasks;
    ::String name;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(tasks);
        fun(name);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(tasks);
        fun(name);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("tasks", tasks);
        fun("name", name);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("tasks", tasks);
        fun("name", name);
    }

    bool operator==(const ::adsfi::SceneDag& t) const
    {
        return (id == t.id) && (tasks == t.tasks) && (name == t.name);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_SCENEDAG_H
