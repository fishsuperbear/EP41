/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HM_IMPL_TYPE_TRANSITION_H
#define HOZON_HM_IMPL_TYPE_TRANSITION_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"

namespace hozon {
namespace hm {
struct Transition {
    ::uint32_t checkpointSrcId;
    ::uint32_t checkpointDestId;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(checkpointSrcId);
        fun(checkpointDestId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(checkpointSrcId);
        fun(checkpointDestId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("checkpointSrcId", checkpointSrcId);
        fun("checkpointDestId", checkpointDestId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("checkpointSrcId", checkpointSrcId);
        fun("checkpointDestId", checkpointDestId);
    }

    bool operator==(const ::hozon::hm::Transition& t) const
    {
        return (checkpointSrcId == t.checkpointSrcId) && (checkpointDestId == t.checkpointDestId);
    }
};
} // namespace hm
} // namespace hozon


#endif // HOZON_HM_IMPL_TYPE_TRANSITION_H
