/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_VO_IMPL_TYPE_CHNINFO_H
#define MDC_VO_IMPL_TYPE_CHNINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"
#include "mdc/vo/impl_type_slotid.h"
#include "impl_type_string.h"
#include "mdc/vo/impl_type_videoparam.h"

namespace mdc {
namespace vo {
struct ChnInfo {
    ::uint8_t chnId;
    ::mdc::vo::SlotId slotId;
    bool isEnabled;
    ::String function;
    ::mdc::vo::VideoParam videoParam;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(chnId);
        fun(slotId);
        fun(isEnabled);
        fun(function);
        fun(videoParam);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(chnId);
        fun(slotId);
        fun(isEnabled);
        fun(function);
        fun(videoParam);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("chnId", chnId);
        fun("slotId", slotId);
        fun("isEnabled", isEnabled);
        fun("function", function);
        fun("videoParam", videoParam);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("chnId", chnId);
        fun("slotId", slotId);
        fun("isEnabled", isEnabled);
        fun("function", function);
        fun("videoParam", videoParam);
    }

    bool operator==(const ::mdc::vo::ChnInfo& t) const
    {
        return (chnId == t.chnId) && (slotId == t.slotId) && (isEnabled == t.isEnabled) && (function == t.function) && (videoParam == t.videoParam);
    }
};
} // namespace vo
} // namespace mdc


#endif // MDC_VO_IMPL_TYPE_CHNINFO_H
