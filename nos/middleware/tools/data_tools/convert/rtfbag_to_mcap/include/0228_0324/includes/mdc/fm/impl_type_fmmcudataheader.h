/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_FMMCUDATAHEADER_H
#define MDC_FM_IMPL_TYPE_FMMCUDATAHEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"
#include "impl_type_uint32_t.h"
#include "impl_type_uint64_t.h"

namespace mdc {
namespace fm {
struct FmMcuDataHeader {
    ::uint8_t type;
    ::uint8_t version;
    ::uint32_t checkSum;
    ::uint32_t seqId;
    ::uint64_t sendTime;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(type);
        fun(version);
        fun(checkSum);
        fun(seqId);
        fun(sendTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(type);
        fun(version);
        fun(checkSum);
        fun(seqId);
        fun(sendTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("type", type);
        fun("version", version);
        fun("checkSum", checkSum);
        fun("seqId", seqId);
        fun("sendTime", sendTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("type", type);
        fun("version", version);
        fun("checkSum", checkSum);
        fun("seqId", seqId);
        fun("sendTime", sendTime);
    }

    bool operator==(const ::mdc::fm::FmMcuDataHeader& t) const
    {
        return (type == t.type) && (version == t.version) && (checkSum == t.checkSum) && (seqId == t.seqId) && (sendTime == t.sendTime);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_FMMCUDATAHEADER_H
