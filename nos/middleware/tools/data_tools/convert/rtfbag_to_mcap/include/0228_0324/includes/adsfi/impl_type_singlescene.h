/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_SINGLESCENE_H
#define ADSFI_IMPL_TYPE_SINGLESCENE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "adsfi/impl_type_scenedag.h"

namespace adsfi {
struct SingleScene {
    ::UInt32 id;
    ::adsfi::SceneDag dag;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(dag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(dag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("dag", dag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("dag", dag);
    }

    bool operator==(const ::adsfi::SingleScene& t) const
    {
        return (id == t.id) && (dag == t.dag);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_SINGLESCENE_H
