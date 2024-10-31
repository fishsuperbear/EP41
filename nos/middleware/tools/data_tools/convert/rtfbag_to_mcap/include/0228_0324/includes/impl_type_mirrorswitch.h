/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_MIRRORSWITCH_H
#define IMPL_TYPE_MIRRORSWITCH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct MirrorSwitch {
    ::UInt8 rearFlViewMirrorSwitch;
    ::UInt8 rearFrViewMirrorSwitch;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(rearFlViewMirrorSwitch);
        fun(rearFrViewMirrorSwitch);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(rearFlViewMirrorSwitch);
        fun(rearFrViewMirrorSwitch);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("rearFlViewMirrorSwitch", rearFlViewMirrorSwitch);
        fun("rearFrViewMirrorSwitch", rearFrViewMirrorSwitch);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("rearFlViewMirrorSwitch", rearFlViewMirrorSwitch);
        fun("rearFrViewMirrorSwitch", rearFrViewMirrorSwitch);
    }

    bool operator==(const ::MirrorSwitch& t) const
    {
        return (rearFlViewMirrorSwitch == t.rearFlViewMirrorSwitch) && (rearFrViewMirrorSwitch == t.rearFrViewMirrorSwitch);
    }
};


#endif // IMPL_TYPE_MIRRORSWITCH_H
