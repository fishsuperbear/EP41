/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_TRAJECTORYPOINTVECTOR_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_TRAJECTORYPOINTVECTOR_SOC_MCU_H
#include "ara/core/vector.h"
#include "hozon/soc_mcu/impl_type_trajectorypoint_soc_mcu.h"

namespace hozon {
namespace soc_mcu {
using TrajectoryPointVector_soc_mcu = ara::core::Vector<hozon::soc_mcu::TrajectoryPoint_soc_mcu>;
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_TRAJECTORYPOINTVECTOR_SOC_MCU_H