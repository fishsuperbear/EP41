/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 * Description: get VCC info and use to transfer driver client event mode
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_EVENTPROXYCALL_H
#define VRTF_VCC_EVENTPROXYCALL_H
#include "vrtf/vcc/api/types.h"
namespace vrtf {
namespace vcc {
class EventProxyCall {
public:
    EventProxyCall() = default;
    virtual ~EventProxyCall() = default;
    virtual void OnDataAvailable(const api::types::EventMsg &eventMsg) = 0;
    virtual void OnDataAvailableDirect(const api::types::EventMsg &eventMsg) = 0;
};
}
}

#endif /* VRTF_VCC_EVENTPROXYCALL_H */
