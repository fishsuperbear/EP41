/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Adapter layer between Ros and Vcc Proxy
 * Create: 2020-04-22
 */
#ifndef RTF_COM_ADAPTER_ROS_PROXY_DIRECT_H
#define RTF_COM_ADAPTER_ROS_PROXY_DIRECT_H

#include <memory>
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/api/recv_buffer.h"

namespace rtf {
namespace com {
namespace adapter {
using RecvMemory = vrtf::vcc::api::types::RecvBuffer;
using SampleInfo = vrtf::vcc::api::types::SampleInfo;
class RosProxyDirect {
public:
    virtual void RosEventDirectCallback(const vrtf::vcc::api::types::EventMsg& eventMsg) = 0;
};
} // namspace adapter
} // namspace com
} // namspace rtf
#endif // RTF_COM_ADAPTER_ROS_PROXY_DIRECT_H