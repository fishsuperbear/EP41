/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_FMMCUDATARETINFO_H
#define MDC_FM_IMPL_TYPE_FMMCUDATARETINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"
#include "impl_type_uint32_t.h"

namespace mdc {
namespace fm {
struct FmMcuDataRetInfo {
    ::uint8_t type;
    ::uint32_t seqId;
    ::uint32_t retCode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(seqId);
        fun(retCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(seqId);
        fun(retCode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("seqId", seqId);
        fun("retCode", retCode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("seqId", seqId);
        fun("retCode", retCode);
    }

    bool operator==(const ::mdc::fm::FmMcuDataRetInfo& t) const
    {
        return (type == t.type) && (seqId == t.seqId) && (retCode == t.retCode);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_FMMCUDATARETINFO_H
