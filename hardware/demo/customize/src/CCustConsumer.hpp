/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CCONSUMER_HPP
#define CCONSUMER_HPP

#include "nvscibuf.h"
#include "CClientCommon.hpp"
#include <atomic>

class CCustConsumer : public CClientCommon
{
  public:
    /** @brief Default constructor. */
    CCustConsumer() = delete;
    CCustConsumer(std::string name, NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle);
    /** @brief Default destructor. */
    virtual ~CCustConsumer() = default;

    // Streaming functions
    NvSciStreamBlock GetQueueHandle(void);
    void SetConsumerConfig(const ConsumerConfig &consConfig);

  protected:
    virtual SIPLStatus HandleClientInit() override;
    SIPLStatus HandlePayload(void) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence);
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex);
    virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus SetUnusedElement(uint32_t elementId) override;
    // Setup element attributes supported by the producer.
    virtual void setEndpointBufAttr(NvSciBufAttrList attrList) override;
    SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus HandleSetupComplete(void)
    {
        printf("CCustConsumer::HandleSetupComplete\n");
        return NVSIPL_STATUS_OK;
    }

    uint32_t m_frameNum = 0U;
    ConsumerConfig m_consConfig;

  private:
    NvSciStreamBlock m_queueHandle = 0U;
    /* Virtual address for the meta buffer */
};
#endif

