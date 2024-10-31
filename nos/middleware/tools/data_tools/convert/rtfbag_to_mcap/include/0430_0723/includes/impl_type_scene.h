/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_SCENE_H
#define IMPL_TYPE_SCENE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_scenendag.h"

struct Scene {
    ::UInt32 id;
    ::ScenenDag sceneDag;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(sceneDag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(sceneDag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("sceneDag", sceneDag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("sceneDag", sceneDag);
    }

    bool operator==(const ::Scene& t) const
    {
        return (id == t.id) && (sceneDag == t.sceneDag);
    }
};


#endif // IMPL_TYPE_SCENE_H
