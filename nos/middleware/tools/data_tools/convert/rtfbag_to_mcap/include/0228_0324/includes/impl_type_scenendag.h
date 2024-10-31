/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_SCENENDAG_H
#define IMPL_TYPE_SCENENDAG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_taskvector.h"
#include "impl_type_string.h"

struct ScenenDag {
    ::UInt32 sceneId;
    ::TaskVector tasks;
    ::String sceneName;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(sceneId);
        fun(tasks);
        fun(sceneName);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(sceneId);
        fun(tasks);
        fun(sceneName);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("sceneId", sceneId);
        fun("tasks", tasks);
        fun("sceneName", sceneName);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("sceneId", sceneId);
        fun("tasks", tasks);
        fun("sceneName", sceneName);
    }

    bool operator==(const ::ScenenDag& t) const
    {
        return (sceneId == t.sceneId) && (tasks == t.tasks) && (sceneName == t.sceneName);
    }
};


#endif // IMPL_TYPE_SCENENDAG_H
