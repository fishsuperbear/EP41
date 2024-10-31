/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_DIAGCOMCLIENT_COMMON_H
#define ARA_DIAG_DIAGCOMCLIENT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/diag/impl_type_requesttriggertype.h"
#include "impl_type_string.h"
#include "impl_type_uint64.h"
#include "impl_type_boolean.h"
#include "ara/diag/impl_type_bytearray.h"
#include "ara/diag/impl_type_bytevector.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace diag {
namespace methods {
namespace GetRequest {
struct Output {
    ::Boolean isResponsePositive;
    ::ara::diag::ByteArray responseData;

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
} // namespace GetRequest
} // namespace methods

class DiagComClient {
public:
    constexpr DiagComClient() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServices/DiagComClient");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace diag
} // namespace ara

#endif // ARA_DIAG_DIAGCOMCLIENT_COMMON_H
