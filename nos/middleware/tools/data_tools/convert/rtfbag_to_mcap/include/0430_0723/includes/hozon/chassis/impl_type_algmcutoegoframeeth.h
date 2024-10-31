/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGMCUTOEGOFRAMEETH_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGMCUTOEGOFRAMEETH_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_algmcuegomemmsg.h"

namespace hozon {
namespace chassis {
struct AlgMcuToEgoFrameEth {
    ::hozon::common::CommonHeader header;
    ::AlgMcuEgoMemMsg msg_mcu_mem;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(msg_mcu_mem);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(msg_mcu_mem);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("msg_mcu_mem", msg_mcu_mem);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("msg_mcu_mem", msg_mcu_mem);
    }

    bool operator==(const ::hozon::chassis::AlgMcuToEgoFrameEth& t) const
    {
        return (header == t.header) && (msg_mcu_mem == t.msg_mcu_mem);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGMCUTOEGOFRAMEETH_H
