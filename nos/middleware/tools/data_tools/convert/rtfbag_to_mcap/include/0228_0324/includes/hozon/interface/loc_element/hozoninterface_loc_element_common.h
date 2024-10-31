/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LOC_ELEMENT_HOZONINTERFACE_LOC_ELEMENT_COMMON_H
#define HOZON_INTERFACE_LOC_ELEMENT_HOZONINTERFACE_LOC_ELEMENT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/locelement/impl_type_locelementframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace loc_element {

class HozonInterface_Loc_Element {
public:
    constexpr HozonInterface_Loc_Element() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Loc_Element");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace loc_element
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LOC_ELEMENT_HOZONINTERFACE_LOC_ELEMENT_COMMON_H
