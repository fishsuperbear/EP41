/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 * Description: This provide the interface of configuration read function.
 * Create: 2022-04-19
 */
#ifndef RTFCM_DFX_CONFIG_INTERFACE_H
#define RTFCM_DFX_CONFIG_INTERFACE_H
#include "ara/com/types.h"
#include "vrtf/driver/dds/dds_driver_types.h"
namespace rtf {
namespace rtfcm {
namespace config {
class DDSDfxConfigInterface {
public:
    virtual ~DDSDfxConfigInterface() = default;
    virtual bool ReadEventData(const ara::com::internal::BindIndex& index, vrtf::driver::dds::DDSEventInfo& data) = 0;
    virtual bool ReadMethodData(const ara::com::internal::BindIndex& index, vrtf::driver::dds::DDSMethodInfo& data) = 0;
    virtual bool ReadServiceDiscoveryData(const ara::com::internal::BindIndex& index, vrtf::driver::dds::DDSServiceDiscoveryInfo& data) = 0;
};
}
}
}
#endif
