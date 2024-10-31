/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SAMM_IMPL_TYPE_SAMMFEATURE_H
#define ARA_SAMM_IMPL_TYPE_SAMMFEATURE_H
#include <cfloat>
#include <cmath>
#include "impl_type_defaultfeaturevector.h"
#include "ara/selectpoint/impl_type_header.h"
#include "impl_type_customfeaturevector.h"

namespace ara {
namespace samm {
struct SammFeature {
    ::DefaultFeatureVector defaultFeatures;
    ::ara::selectpoint::Header header;
    ::CustomFeatureVector customFeatures;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(defaultFeatures);
        fun(header);
        fun(customFeatures);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(defaultFeatures);
        fun(header);
        fun(customFeatures);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("defaultFeatures", defaultFeatures);
        fun("header", header);
        fun("customFeatures", customFeatures);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("defaultFeatures", defaultFeatures);
        fun("header", header);
        fun("customFeatures", customFeatures);
    }

    bool operator==(const ::ara::samm::SammFeature& t) const
    {
        return (defaultFeatures == t.defaultFeatures) && (header == t.header) && (customFeatures == t.customFeatures);
    }
};
} // namespace samm
} // namespace ara


#endif // ARA_SAMM_IMPL_TYPE_SAMMFEATURE_H
