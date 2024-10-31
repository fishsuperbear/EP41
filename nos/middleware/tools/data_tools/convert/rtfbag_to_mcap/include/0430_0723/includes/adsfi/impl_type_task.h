/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_TASK_H
#define ADSFI_IMPL_TYPE_TASK_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "adsfi/impl_type_jobvector.h"

namespace adsfi {
struct Task {
    ::String name;
    ::adsfi::JobVector jobList;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(name);
        fun(jobList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(name);
        fun(jobList);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("name", name);
        fun("jobList", jobList);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("name", name);
        fun("jobList", jobList);
    }

    bool operator==(const ::adsfi::Task& t) const
    {
        return (name == t.name) && (jobList == t.jobList);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_TASK_H
