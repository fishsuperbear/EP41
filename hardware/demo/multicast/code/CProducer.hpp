/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CPRODUCER_HPP
#define CPRODUCER_HPP

#include <atomic>

#include "nvscibuf.h"
#include "CClientCommon.hpp"

class CProducer: public CClientCommon
{
public:

    /** @brief Default constructor. */
    CProducer() = delete;
    /** @brief Default destructor. */
    CProducer(std::string name, NvSciStreamBlock handle, uint32_t uSensor);
    virtual ~CProducer() = default;
    SIPLStatus Post(void *pBuffer);
protected:
    virtual SIPLStatus HandleStreamInit(void) override;
    virtual SIPLStatus HandleSetupComplete(void) override;
    virtual void OnPacketGotten(uint32_t packetIndex) = 0;
    virtual SIPLStatus HandlePayload(void) override;
    virtual SIPLStatus MapPayload(void *pBuffer, uint32_t& packetIndex) = 0;
    virtual SIPLStatus GetPostfence(uint32_t packetIndex, NvSciSyncFence *pPostfence) = 0;
    virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) override;

    uint32_t m_numConsumers;
private:
    std::atomic<uint32_t> m_numBuffersWithConsumer;
};
#endif

