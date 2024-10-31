/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SAMM_IMPL_TYPE_FEATURE_H
#define ARA_SAMM_IMPL_TYPE_FEATURE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_string.h"

namespace ara {
namespace samm {
struct Feature {
    ::UInt32 id;
    ::String name;
    ::String description;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(name);
        fun(description);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(name);
        fun(description);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("name", name);
        fun("description", description);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("name", name);
        fun("description", description);
    }

    bool operator==(const ::ara::samm::Feature& t) const
    {
        return (id == t.id) && (name == t.name) && (description == t.description);
    }
};
} // namespace samm
} // namespace ara


#endif // ARA_SAMM_IMPL_TYPE_FEATURE_H
