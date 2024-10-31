/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_FMPOSTPROCESSSERVICE_COMMON_H
#define HOZON_FM_FMPOSTPROCESSSERVICE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/fm/impl_type_faultclusterdata.h"
#include "impl_type_string.h"
#include "impl_type_stringvector.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace fm {
namespace methods {
} // namespace methods

class FmPostProcessService {
public:
    constexpr FmPostProcessService() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/Service/FmPostProcessService");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace fm
} // namespace hozon

#endif // HOZON_FM_FMPOSTPROCESSSERVICE_COMMON_H
