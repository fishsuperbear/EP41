/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_DIAGCOMNOTIFIER_COMMON_H
#define ARA_DIAG_DIAGCOMNOTIFIER_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/diag/impl_type_bytevector.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace diag {

class DiagComNotifier {
public:
    constexpr DiagComNotifier() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformService/DiagComNotifier");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace diag
} // namespace ara

#endif // ARA_DIAG_DIAGCOMNOTIFIER_COMMON_H
