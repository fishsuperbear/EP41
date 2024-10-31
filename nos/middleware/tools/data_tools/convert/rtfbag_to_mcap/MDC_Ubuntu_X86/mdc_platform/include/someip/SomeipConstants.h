/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 */

#ifndef SOMEIP_CONSTANTS_H
#define SOMEIP_CONSTANTS_H

#include <someip/SomeipTypes.h>

namespace Someip {
const MajorVersion DEFAULT_MAJOR_VERSION{0U};
const MinorVersion DEFAULT_MINOR_VERSION{0U};
const uint16_t USE_ILLEGAL_PORT{0xFFFFU};
const ServiceID ANY_SERVICE_ID{0xFFFFU};
const InstanceID ANY_INSTANCE_ID{0xFFFFU};
const MethodID ANY_METHOD_ID{0xFFFFU};
const EventID ANY_EVENT_ID{0xFFFFU};
const MajorVersion ANY_MAJOR_VERSION{0xFFU};
const MinorVersion ANY_MINOR_VERSION{0xFFFFFFFFU};
const ClientID ILLEGAL_CLIENT_ID{0U};
const ClientID SOMEIP_ROUTING_CLIENT{0U};
} // namespace Someip

#endif  // SOMEIP_CONSTANTS_H
