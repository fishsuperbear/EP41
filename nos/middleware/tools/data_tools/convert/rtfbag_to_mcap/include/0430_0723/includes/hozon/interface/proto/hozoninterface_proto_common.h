/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_COMMON_H
#define HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_string.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace proto {
namespace methods {
namespace ProtoMethod_ReqWithResp {
struct Output {
    ::String output;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(output);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(output);
    }

    bool operator==(const Output& t) const
    {
       return (output == t.output);
    }
};
} // namespace ProtoMethod_ReqWithResp
} // namespace methods

class HozonInterface_Proto {
public:
    constexpr HozonInterface_Proto() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Proto");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace proto
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_COMMON_H
