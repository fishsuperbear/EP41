/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_EGOTRAJECTORY_EGOTRAJECTORYSERVICEINTERFACE_COMMON_H
#define ARA_EGOTRAJECTORY_EGOTRAJECTORYSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/egotrajectory/impl_type_egotrajectory.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace egotrajectory {

class EgoTrajectoryServiceInterface {
public:
    constexpr EgoTrajectoryServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/EgoTrajectoryServiceInterface/EgoTrajectoryServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace egotrajectory
} // namespace ara

#endif // ARA_EGOTRAJECTORY_EGOTRAJECTORYSERVICEINTERFACE_COMMON_H
