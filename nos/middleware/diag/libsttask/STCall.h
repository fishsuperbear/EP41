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
 * @file  STCall.h
 * @brief Class of STCall
 */
#ifndef STCALL_H
#define STCALL_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <string>
#include "STObjectDef.h"

namespace hozon {
namespace netaos {
namespace sttask {
    /**
     * @brief Class of STCall
     *
     * This class represent an object in a sync call.
     */
    class STCall
    {
    public:
        STCall(uint32_t callKind, uint32_t callId = 0);
        virtual ~STCall();

        uint32_t              getCallKind() const;
        uint32_t              getCallId() const;

        virtual std::string    toString();

    private:
        STCall(const STCall&);
        STCall& operator=(const STCall&);
        const uint32_t m_callKind;
        const uint32_t m_callId;
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* u */
/* EOF */