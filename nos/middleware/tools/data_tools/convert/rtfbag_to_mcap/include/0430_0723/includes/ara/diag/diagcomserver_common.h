/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_DIAGCOMSERVER_COMMON_H
#define ARA_DIAG_DIAGCOMSERVER_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/diag/impl_type_bytevector.h"
#include "impl_type_boolean.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace diag {
namespace methods {
namespace Request {
struct Output {
    ::Boolean isResponsePositive;
    ::ara::diag::ByteVector responseData;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isResponsePositive);
        fun(responseData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isResponsePositive);
        fun(responseData);
    }

    bool operator==(const Output& t) const
    {
       return (isResponsePositive == t.isResponsePositive) && (responseData == t.responseData);
    }
};
} // namespace Request
} // namespace methods

class DiagComServer {
public:
    constexpr DiagComServer() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServices/DiagComServer");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace diag
} // namespace ara

#endif // ARA_DIAG_DIAGCOMSERVER_COMMON_H
