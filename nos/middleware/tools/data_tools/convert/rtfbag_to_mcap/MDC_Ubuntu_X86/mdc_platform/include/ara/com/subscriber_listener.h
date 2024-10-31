/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This provide the interface of SubscriberListener.
 * Create: 2022-06-18
 */

#ifndef ARA_COM_SUBSCRIBER_LISTENER_H
#define ARA_COM_SUBSCRIBER_LISTENER_H

#include "vrtf/vcc/api/subscriber_listener.h"

namespace ara {
namespace com {
using SubscriberListener = vrtf::vcc::api::types::SubscriberListener;
using ListenerTimeOutInfo = vrtf::vcc::api::types::ListenerTimeoutInfo;
}
}

#endif //ARA_COM_SUBSCRIBER_LISTENER_H
