/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_ESTOP_H
#define HOZON_PLANNING_IMPL_TYPE_ESTOP_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "hozon/common/impl_type_commonheader.h"

namespace hozon {
namespace planning {
struct Estop {
    ::UInt8 isStop;
    ::String description;
    ::hozon::common::CommonHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isStop);
        fun(description);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isStop);
        fun(description);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("isStop", isStop);
        fun("description", description);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("isStop", isStop);
        fun("description", description);
        fun("header", header);
    }

    bool operator==(const ::hozon::planning::Estop& t) const
    {
        return (isStop == t.isStop) && (description == t.description) && (header == t.header);
    }
};
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_ESTOP_H
