/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_VO_IMPL_TYPE_VIDEOPARAM_H
#define MDC_VO_IMPL_TYPE_VIDEOPARAM_H
#include <cfloat>
#include <cmath>
#include "ara/display/impl_type_imageformattypee.h"
#include "impl_type_uint32_t.h"
#include "mdc/vo/impl_type_resolution.h"

namespace mdc {
namespace vo {
struct VideoParam {
    ::ara::display::ImageFormatTypeE imageFormat;
    ::uint32_t frameRate;
    ::mdc::vo::Resolution resolution;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(imageFormat);
        fun(frameRate);
        fun(resolution);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(imageFormat);
        fun(frameRate);
        fun(resolution);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("imageFormat", imageFormat);
        fun("frameRate", frameRate);
        fun("resolution", resolution);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("imageFormat", imageFormat);
        fun("frameRate", frameRate);
        fun("resolution", resolution);
    }

    bool operator==(const ::mdc::vo::VideoParam& t) const
    {
        return (imageFormat == t.imageFormat) && (frameRate == t.frameRate) && (resolution == t.resolution);
    }
};
} // namespace vo
} // namespace mdc


#endif // MDC_VO_IMPL_TYPE_VIDEOPARAM_H
