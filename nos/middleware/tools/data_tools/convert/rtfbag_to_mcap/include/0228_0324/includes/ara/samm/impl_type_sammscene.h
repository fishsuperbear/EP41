/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SAMM_IMPL_TYPE_SAMMSCENE_H
#define ARA_SAMM_IMPL_TYPE_SAMMSCENE_H
#include <cfloat>
#include <cmath>
#include "impl_type_scene.h"
#include "ara/samm/impl_type_header.h"

namespace ara {
namespace samm {
struct SammScene {
    ::Scene scene;
    ::ara::samm::Header header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(scene);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(scene);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("scene", scene);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("scene", scene);
        fun("header", header);
    }

    bool operator==(const ::ara::samm::SammScene& t) const
    {
        return (scene == t.scene) && (header == t.header);
    }
};
} // namespace samm
} // namespace ara


#endif // ARA_SAMM_IMPL_TYPE_SAMMSCENE_H
