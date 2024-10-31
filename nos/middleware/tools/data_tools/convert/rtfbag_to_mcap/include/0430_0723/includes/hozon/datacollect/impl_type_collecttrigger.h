/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DATACOLLECT_IMPL_TYPE_COLLECTTRIGGER_H
#define HOZON_DATACOLLECT_IMPL_TYPE_COLLECTTRIGGER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace hozon {
namespace datacollect {
struct CollectTrigger {
    ::UInt32 trigger_id;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(trigger_id);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(trigger_id);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("trigger_id", trigger_id);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("trigger_id", trigger_id);
    }

    bool operator==(const ::hozon::datacollect::CollectTrigger& t) const
    {
        return (trigger_id == t.trigger_id);
    }
};
} // namespace datacollect
} // namespace hozon


#endif // HOZON_DATACOLLECT_IMPL_TYPE_COLLECTTRIGGER_H
