/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_TASKDESCRIPTION_H
#define IMPL_TYPE_TASKDESCRIPTION_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_jobvector.h"

struct TaskDescription {
    ::String taskName;
    ::JobVector jobList;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(taskName);
        fun(jobList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(taskName);
        fun(jobList);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("taskName", taskName);
        fun("jobList", jobList);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("taskName", taskName);
        fun("jobList", jobList);
    }

    bool operator==(const ::TaskDescription& t) const
    {
        return (taskName == t.taskName) && (jobList == t.jobList);
    }
};


#endif // IMPL_TYPE_TASKDESCRIPTION_H