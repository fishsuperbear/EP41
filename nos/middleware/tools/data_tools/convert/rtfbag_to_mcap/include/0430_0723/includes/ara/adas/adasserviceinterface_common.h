/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADAS_ADASSERVICEINTERFACE_COMMON_H
#define ARA_ADAS_ADASSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/vehicle/impl_type_flcfr01info.h"
#include "ara/vehicle/impl_type_flcfr02info.h"
#include "ara/vehicle/impl_type_flrfr01info.h"
#include "ara/vehicle/impl_type_flrfr02info.h"
#include "ara/vehicle/impl_type_flrfr03info.h"
#include "ara/vehicle/impl_type_apainfo.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace adas {

class AdasServiceInterface {
public:
    constexpr AdasServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/AdasServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace adas
} // namespace ara

#endif // ARA_ADAS_ADASSERVICEINTERFACE_COMMON_H
