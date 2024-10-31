/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_OCR_HOZONINTERFACE_OCR_COMMON_H
#define HOZON_INTERFACE_OCR_HOZONINTERFACE_OCR_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/ocr/impl_type_algocrframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace ocr {

class HozonInterface_Ocr {
public:
    constexpr HozonInterface_Ocr() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Ocr");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace ocr
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_OCR_HOZONINTERFACE_OCR_COMMON_H
