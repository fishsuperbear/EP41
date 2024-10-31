/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_VERSIONISP_H
#define MDC_DEVM_IMPL_TYPE_VERSIONISP_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace mdc {
namespace devm {
struct VersionISP {
    ::String uboot;
    ::String os;
    ::String sys;
    ::String ubootBak;
    ::String osBak;
    ::String sysBak;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(uboot);
        fun(os);
        fun(sys);
        fun(ubootBak);
        fun(osBak);
        fun(sysBak);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(uboot);
        fun(os);
        fun(sys);
        fun(ubootBak);
        fun(osBak);
        fun(sysBak);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("uboot", uboot);
        fun("os", os);
        fun("sys", sys);
        fun("ubootBak", ubootBak);
        fun("osBak", osBak);
        fun("sysBak", sysBak);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("uboot", uboot);
        fun("os", os);
        fun("sys", sys);
        fun("ubootBak", ubootBak);
        fun("osBak", osBak);
        fun("sysBak", sysBak);
    }

    bool operator==(const ::mdc::devm::VersionISP& t) const
    {
        return (uboot == t.uboot) && (os == t.os) && (sys == t.sys) && (ubootBak == t.ubootBak) && (osBak == t.osBak) && (sysBak == t.sysBak);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_VERSIONISP_H
