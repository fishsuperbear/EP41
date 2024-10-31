/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CCONSUMER_HPP
#define CCONSUMER_HPP

#include "nvscibuf.h"
#include "sensor/nvs_consumer/CClientCommon.hpp"
#include <atomic>

namespace hozon {
namespace netaos {
namespace desay { 

class CConsumer: public CClientCommon
{
public:

    /** @brief Default constructor. */
    CConsumer() = delete;
    CConsumer(std::string name, NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle);
    /** @brief Default destructor. */
    virtual ~CConsumer() = default;

    // Streaming functions
    NvSciStreamBlock GetQueueHandle(void);
    void SetConsumerConfig(const ConsumerConfig& consConfig);
    void SetDisplayConfig(const DisplayConfig& dispConfig);

protected:
    SIPLStatus HandlePayload(void) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) = 0;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) = 0;
    virtual bool ToSkipFrame(uint32_t frameNum) {return false;};
    virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus SetUnusedElement(uint32_t elementId) override;

    uint32_t m_frameNum = 0U;
    ConsumerConfig m_consConfig;
    DisplayConfig m_dispConfig;

private:
    NvSciStreamBlock m_queueHandle = 0U;
    /* Virtual address for the meta buffer */
};

}
}
}

#endif

