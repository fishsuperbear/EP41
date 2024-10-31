/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_IMPL_TYPE_COMCTRLREQUESTPARAMSTYPE_H
#define ARA_DIAG_IMPL_TYPE_COMCTRLREQUESTPARAMSTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"

namespace ara {
namespace diag {
struct ComCtrlRequestParamsType {
    ::UInt8 controlType;
    ::UInt8 communicationType;
    ::UInt8 subnetNumber;
    ::UInt16 nodeIdentificationNumber;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(controlType);
        fun(communicationType);
        fun(subnetNumber);
        fun(nodeIdentificationNumber);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(controlType);
        fun(communicationType);
        fun(subnetNumber);
        fun(nodeIdentificationNumber);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("controlType", controlType);
        fun("communicationType", communicationType);
        fun("subnetNumber", subnetNumber);
        fun("nodeIdentificationNumber", nodeIdentificationNumber);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("controlType", controlType);
        fun("communicationType", communicationType);
        fun("subnetNumber", subnetNumber);
        fun("nodeIdentificationNumber", nodeIdentificationNumber);
    }

    bool operator==(const ::ara::diag::ComCtrlRequestParamsType& t) const
    {
        return (controlType == t.controlType) && (communicationType == t.communicationType) && (subnetNumber == t.subnetNumber) && (nodeIdentificationNumber == t.nodeIdentificationNumber);
    }
};
} // namespace diag
} // namespace ara


#endif // ARA_DIAG_IMPL_TYPE_COMCTRLREQUESTPARAMSTYPE_H
