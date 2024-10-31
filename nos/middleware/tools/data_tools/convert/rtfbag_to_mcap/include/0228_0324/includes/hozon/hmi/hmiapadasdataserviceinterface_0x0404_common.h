/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_COMMON_H
#define HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi/impl_type_apa_dataproperties_struct.h"
#include "hozon/hmi/impl_type_hpp_path_struct.h"
#include "hozon/hmi/impl_type_nns_info_struct.h"
#include "hozon/hmi/impl_type_ins_info_struct.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {

class HmiAPADASdataServiceInterface_0x0404 {
public:
    constexpr HmiAPADASdataServiceInterface_0x0404() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HmiAPADASdataServiceInterface_0x0404");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmi
} // namespace hozon

#endif // HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_COMMON_H
