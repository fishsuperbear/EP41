/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_HZSOCMCUAEBRECORDCLIENTSERVICEINTERFACE_COMMON_H
#define HOZON_SOC_MCU_HZSOCMCUAEBRECORDCLIENTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/soc2mcu/impl_type_aebfcw_datarecordframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace soc_mcu {

class HzSocMcuAebRecordClientServiceInterface {
public:
    constexpr HzSocMcuAebRecordClientServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HzSocMcuAebRecordClientServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace soc_mcu
} // namespace hozon

#endif // HOZON_SOC_MCU_HZSOCMCUAEBRECORDCLIENTSERVICEINTERFACE_COMMON_H
