/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_AVP_MAP_HOZONINTERFACE_AVP_MAP_COMMON_H
#define HOZON_INTERFACE_AVP_MAP_HOZONINTERFACE_AVP_MAP_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/avpmapmsg/impl_type_avpmapmsgframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace avp_map {

class HozonInterface_Avp_Map {
public:
    constexpr HozonInterface_Avp_Map() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/ServiceInterface/HozonInterface_Avp_Map");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace avp_map
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_AVP_MAP_HOZONINTERFACE_AVP_MAP_COMMON_H
