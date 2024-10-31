/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_IMPL_TYPE_GIDSTATUSSTRUCT_H
#define ARA_DIAG_IMPL_TYPE_GIDSTATUSSTRUCT_H
#include <cfloat>
#include <cmath>
#include "ara/diag/impl_type_bytearray.h"
#include "impl_type_uint8.h"

namespace ara {
namespace diag {
struct GidStatusStruct {
    ::ara::diag::ByteArray GID;
    ::UInt8 furtherActionReq;
    ::UInt8 syncStatus;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(GID);
        fun(furtherActionReq);
        fun(syncStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(GID);
        fun(furtherActionReq);
        fun(syncStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("GID", GID);
        fun("furtherActionReq", furtherActionReq);
        fun("syncStatus", syncStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("GID", GID);
        fun("furtherActionReq", furtherActionReq);
        fun("syncStatus", syncStatus);
    }

    bool operator==(const ::ara::diag::GidStatusStruct& t) const
    {
        return (GID == t.GID) && (furtherActionReq == t.furtherActionReq) && (syncStatus == t.syncStatus);
    }
};
} // namespace diag
} // namespace ara


#endif // ARA_DIAG_IMPL_TYPE_GIDSTATUSSTRUCT_H
