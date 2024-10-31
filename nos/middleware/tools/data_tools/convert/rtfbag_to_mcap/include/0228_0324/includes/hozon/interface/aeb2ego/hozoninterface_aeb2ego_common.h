/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_AEB2EGO_HOZONINTERFACE_AEB2EGO_COMMON_H
#define HOZON_INTERFACE_AEB2EGO_HOZONINTERFACE_AEB2EGO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/chassis/impl_type_aebtoegoinfoframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace aeb2ego {

class HozonInterface_Aeb2Ego {
public:
    constexpr HozonInterface_Aeb2Ego() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Aeb2Ego");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace aeb2ego
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_AEB2EGO_HOZONINTERFACE_AEB2EGO_COMMON_H
