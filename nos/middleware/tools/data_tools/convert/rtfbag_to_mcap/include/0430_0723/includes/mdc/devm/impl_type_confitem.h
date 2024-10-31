/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_CONFITEM_H
#define MDC_DEVM_IMPL_TYPE_CONFITEM_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace mdc {
namespace devm {
struct ConfItem {
    ::String key;
    ::String value;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(key);
        fun(value);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(key);
        fun(value);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("key", key);
        fun("value", value);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("key", key);
        fun("value", value);
    }

    bool operator==(const ::mdc::devm::ConfItem& t) const
    {
        return (key == t.key) && (value == t.value);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_CONFITEM_H
