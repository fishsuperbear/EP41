/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_PLANNINGDEC_HOZONINTERFACE_PLANNINGDECISIONINFO_COMMON_H
#define HOZON_HMI_PLANNINGDEC_HOZONINTERFACE_PLANNINGDECISIONINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi/impl_type_hmihafplanningdecisioninfo.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi_planningdec {

class HozonInterface_PlanningDecisionInfo {
public:
    constexpr HozonInterface_PlanningDecisionInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_PlanningDecisionInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmi_planningdec
} // namespace hozon

#endif // HOZON_HMI_PLANNINGDEC_HOZONINTERFACE_PLANNINGDECISIONINFO_COMMON_H
