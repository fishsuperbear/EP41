/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_CANCTRLCONFIG_H
#define MDC_DEVM_IMPL_TYPE_CANCTRLCONFIG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "mdc/devm/impl_type_canchecklist.h"

namespace mdc {
namespace devm {
struct CanCtrlConfig {
    ::UInt8 channelId;
    ::mdc::devm::CanCheckList upStream;
    ::mdc::devm::CanCheckList downStream;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(channelId);
        fun(upStream);
        fun(downStream);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(channelId);
        fun(upStream);
        fun(downStream);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("channelId", channelId);
        fun("upStream", upStream);
        fun("downStream", downStream);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("channelId", channelId);
        fun("upStream", upStream);
        fun("downStream", downStream);
    }

    bool operator==(const ::mdc::devm::CanCtrlConfig& t) const
    {
        return (channelId == t.channelId) && (upStream == t.upStream) && (downStream == t.downStream);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_CANCTRLCONFIG_H
