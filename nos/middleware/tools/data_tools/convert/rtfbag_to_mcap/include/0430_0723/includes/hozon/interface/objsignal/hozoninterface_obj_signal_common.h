/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_OBJSIGNAL_HOZONINTERFACE_OBJ_SIGNAL_COMMON_H
#define HOZON_INTERFACE_OBJSIGNAL_HOZONINTERFACE_OBJ_SIGNAL_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/object/impl_type_objectsignalframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace objsignal {

class HozonInterface_Obj_Signal {
public:
    constexpr HozonInterface_Obj_Signal() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Obj_Signal");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace objsignal
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_OBJSIGNAL_HOZONINTERFACE_OBJ_SIGNAL_COMMON_H
