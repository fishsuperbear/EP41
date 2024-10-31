/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef SOTIF_CROSSVALIDATEARRAYINTF_COMMON_H
#define SOTIF_CROSSVALIDATEARRAYINTF_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "sotif/impl_type_crossvalidatearray.h"
#include <cfloat>
#include <cmath>

namespace sotif {

class CrossValidateArrayIntf {
public:
    constexpr CrossValidateArrayIntf() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/Sotif/CrossValidateArrayServiceInterface/CrossValidateArrayIntf");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace sotif

#endif // SOTIF_CROSSVALIDATEARRAYINTF_COMMON_H
