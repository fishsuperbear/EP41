/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_EQ3SERVICEINTERFACE_COMMON_H
#define HOZON_EQ3_EQ3SERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/eq3/impl_type_eq3visdatatype.h"
#include "hozon/eq3/impl_type_pedestrianinfos.h"
#include "hozon/eq3/impl_type_rtdisinfos.h"
#include "hozon/eq3/impl_type_rtsdisinfos.h"
#include "hozon/eq3/impl_type_visobsmsgsdatatype.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace eq3 {

class Eq3ServiceInterface {
public:
    constexpr Eq3ServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/Eq3ServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace eq3
} // namespace hozon

#endif // HOZON_EQ3_EQ3SERVICEINTERFACE_COMMON_H
