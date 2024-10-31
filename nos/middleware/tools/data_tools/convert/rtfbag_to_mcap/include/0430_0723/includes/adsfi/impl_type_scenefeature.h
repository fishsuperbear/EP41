/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_SCENEFEATURE_H
#define ADSFI_IMPL_TYPE_SCENEFEATURE_H
#include <cfloat>
#include <cmath>
#include "adsfi/impl_type_featurevector.h"
#include "ara/common/impl_type_commonheader.h"

namespace adsfi {
struct SceneFeature {
    ::adsfi::FeatureVector defaultFeatures;
    ::ara::common::CommonHeader header;
    ::adsfi::FeatureVector customFeatures;

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

    bool operator==(const ::adsfi::SceneFeature& t) const
    {
        return (defaultFeatures == t.defaultFeatures) && (header == t.header) && (customFeatures == t.customFeatures);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_SCENEFEATURE_H
