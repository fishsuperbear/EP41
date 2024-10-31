/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_IMPL_TYPE_DIDINFO_H
#define ARA_DIAG_IMPL_TYPE_DIDINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "ara/diag/impl_type_bytearray.h"

namespace ara {
namespace diag {
struct DidInfo {
    ::UInt16 didId;
    ::ara::diag::ByteArray didValue;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(didId);
        fun(didValue);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(didId);
        fun(didValue);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("didId", didId);
        fun("didValue", didValue);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("didId", didId);
        fun("didValue", didValue);
    }

    bool operator==(const ::ara::diag::DidInfo& t) const
    {
        return (didId == t.didId) && (didValue == t.didValue);
    }
};
} // namespace diag
} // namespace ara


#endif // ARA_DIAG_IMPL_TYPE_DIDINFO_H
