/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_CAMERAVIDEOCOMPLINKOBJ_H
#define MDC_DEVM_IMPL_TYPE_CAMERAVIDEOCOMPLINKOBJ_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_int32array.h"

namespace mdc {
namespace devm {
struct CameraVideoCompLinkObj {
    ::String videoCompName;
    ::Int32Array cameraList;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(videoCompName);
        fun(cameraList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(videoCompName);
        fun(cameraList);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("videoCompName", videoCompName);
        fun("cameraList", cameraList);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("videoCompName", videoCompName);
        fun("cameraList", cameraList);
    }

    bool operator==(const ::mdc::devm::CameraVideoCompLinkObj& t) const
    {
        return (videoCompName == t.videoCompName) && (cameraList == t.cameraList);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_CAMERAVIDEOCOMPLINKOBJ_H
