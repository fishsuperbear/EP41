/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_SCENE_H
#define ADSFI_IMPL_TYPE_SCENE_H
#include <cfloat>
#include <cmath>
#include "adsfi/impl_type_singlescene.h"
#include "ara/common/impl_type_commonheader.h"

namespace adsfi {
struct Scene {
    ::adsfi::SingleScene targetScene;
    ::ara::common::CommonHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(targetScene);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(targetScene);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("targetScene", targetScene);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("targetScene", targetScene);
        fun("header", header);
    }

    bool operator==(const ::adsfi::Scene& t) const
    {
        return (targetScene == t.targetScene) && (header == t.header);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_SCENE_H
