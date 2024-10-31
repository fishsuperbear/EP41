/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_IMPL_TYPE_HZFAULTEVENTTOMCU_H
#define HOZON_FM_IMPL_TYPE_HZFAULTEVENTTOMCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16_t.h"
#include "impl_type_uint8_t.h"
#include "hozon/fm/impl_type_postproccesarray.h"

namespace hozon {
namespace fm {
struct HzFaultEventToMCU {
    ::uint16_t faultId;
    ::uint8_t faultObj;
    ::uint8_t faultStatus;
    ::hozon::fm::PostProccesArray postProcessArray;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(postProcessArray);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultId);
        fun(faultObj);
        fun(faultStatus);
        fun(postProcessArray);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("postProcessArray", postProcessArray);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("faultStatus", faultStatus);
        fun("postProcessArray", postProcessArray);
    }

    bool operator==(const ::hozon::fm::HzFaultEventToMCU& t) const
    {
        return (faultId == t.faultId) && (faultObj == t.faultObj) && (faultStatus == t.faultStatus) && (postProcessArray == t.postProcessArray);
    }
};
} // namespace fm
} // namespace hozon


#endif // HOZON_FM_IMPL_TYPE_HZFAULTEVENTTOMCU_H
