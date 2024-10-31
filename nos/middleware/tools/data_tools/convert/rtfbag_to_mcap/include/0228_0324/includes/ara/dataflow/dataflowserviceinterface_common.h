/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DATAFLOW_DATAFLOWSERVICEINTERFACE_COMMON_H
#define ARA_DATAFLOW_DATAFLOWSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/dataflow/impl_type_dataflowdatatype.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace dataflow {

class DataFlowServiceInterface {
public:
    constexpr DataFlowServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/DataFlowServiceInterface/DataFlowServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace dataflow
} // namespace ara

#endif // ARA_DATAFLOW_DATAFLOWSERVICEINTERFACE_COMMON_H
