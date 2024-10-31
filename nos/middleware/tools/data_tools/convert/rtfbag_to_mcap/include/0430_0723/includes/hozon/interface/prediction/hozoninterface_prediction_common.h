/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PREDICTION_HOZONINTERFACE_PREDICTION_COMMON_H
#define HOZON_INTERFACE_PREDICTION_HOZONINTERFACE_PREDICTION_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/prediction/impl_type_predictionframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace prediction {

class HozonInterface_Prediction {
public:
    constexpr HozonInterface_Prediction() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Prediction");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace prediction
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PREDICTION_HOZONINTERFACE_PREDICTION_COMMON_H
