/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_CANCHECKLIST_H
#define MDC_DEVM_IMPL_TYPE_CANCHECKLIST_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "mdc/devm/impl_type_canidinfolist.h"

namespace mdc {
namespace devm {
struct CanCheckList {
    ::UInt8 idNumber;
    ::UInt8 controlMethod;
    ::mdc::devm::CanIdInfoList canIdInfos;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(idNumber);
        fun(controlMethod);
        fun(canIdInfos);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(idNumber);
        fun(controlMethod);
        fun(canIdInfos);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("idNumber", idNumber);
        fun("controlMethod", controlMethod);
        fun("canIdInfos", canIdInfos);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("idNumber", idNumber);
        fun("controlMethod", controlMethod);
        fun("canIdInfos", canIdInfos);
    }

    bool operator==(const ::mdc::devm::CanCheckList& t) const
    {
        return (idNumber == t.idNumber) && (controlMethod == t.controlMethod) && (canIdInfos == t.canIdInfos);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_CANCHECKLIST_H
