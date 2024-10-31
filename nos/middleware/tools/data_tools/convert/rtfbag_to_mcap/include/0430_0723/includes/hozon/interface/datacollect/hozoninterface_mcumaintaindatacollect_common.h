/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUMAINTAINDATACOLLECT_COMMON_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUMAINTAINDATACOLLECT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/soc_mcu/impl_type_mcudebugdatatype.h"
#include "hozon/soc_mcu/impl_type_mcuclouddatatype.h"
#include "hozon/soc_mcu/impl_type_dtdebug_adas.h"
#include "hozon/soc_mcu/impl_type_struct_soc_mcu_array_record_algo_state.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace datacollect {

class HozonInterface_MCUMaintainDataCollect {
public:
    constexpr HozonInterface_MCUMaintainDataCollect() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_MCUMaintainDataCollect");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUMAINTAINDATACOLLECT_COMMON_H
