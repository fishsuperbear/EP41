/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file STCall.cpp
 * @brief Implements of STCall class
 */

#include "STCall.h"
#include <string.h>

namespace hozon {
namespace netaos {
namespace sttask {

    STCall::STCall(uint32_t callKind, uint32_t callId)
        : m_callKind(callKind)
        , m_callId(callId)
    {
        // Nothing here.
    }

    STCall::~STCall()
    {
        // Nothing here.
    }

    uint32_t STCall::getCallKind() const
    {
        return m_callKind;
    }

    uint32_t STCall::getCallId() const
    {
        return m_callId;
    }

    std::string STCall::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), ("Call[%p, %X, %X]"), this, getCallKind(), getCallId());
        val.assign(buf, strlen(buf));
        return val;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */