/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CCUSTPRODUCER_HPP
#define CCUSTPRODUCER_HPP

#include <atomic>

#include "nvscibuf.h"
#include "CClientCommon.hpp"

class CCustProducer : public CClientCommon
{
  public:
    /** @brief Default constructor. */
    CCustProducer() = delete;
    /** @brief Default destructor. */
    CCustProducer(std::string name, NvSciStreamBlock handle, uint32_t uSensor);
    virtual ~CCustProducer() = default;
    /* virtual SIPLStatus Post(void *pBuffer); */
    SIPLStatus post(uint32_t pIndex);

  protected:
    virtual SIPLStatus HandleClientInit(void) override;
    virtual SIPLStatus HandleStreamInit(void) override;
    virtual SIPLStatus HandleSetupComplete(void) override;
    virtual void OnPacketGotten(uint32_t packetIndex);
    virtual SIPLStatus HandlePayload(void) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus MapPayload(void *pBuffer, uint32_t &packetIndex);
    virtual SIPLStatus GetPostfence(uint32_t packetIndex, NvSciSyncFence *pPostfence)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual SIPLStatus InsertPrefence(PacketElementType userType, uint32_t packetIndex, NvSciSyncFence &prefence)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) override;

    // Setup element attributes supported by the producer.
    virtual void setEndpointBufAttr(NvSciBufAttrList attrList) override;

    uint32_t m_numConsumers;
    std::atomic<uint32_t> m_numBuffersWithConsumer;
    int32_t m_elemTypeToOutputType[MAX_NUM_ELEMENTS];
    PacketElementType m_outputTypeToElemType[MAX_OUTPUTS_PER_SENSOR];
    std::mutex mutexPacketIndexs;
    std::vector<uint32_t> packetIndexs;
  public:
    void insertToPacketIndexs(uint32_t index) {
        std::unique_lock<std::mutex> lock(mutexPacketIndexs);
        packetIndexs.push_back(index);
        printf("insert packet,size=%d\n",packetIndexs.size());
    }
    uint32_t getFromPacketIndexs() {
        std::unique_lock<std::mutex> lock(mutexPacketIndexs);
        printf("get packet,size=%d\n",packetIndexs.size());
        if (!packetIndexs.empty()) {
            uint32_t index = packetIndexs.front();
            packetIndexs.erase(packetIndexs.begin());
            return index;
        } else {
            return -1;
        }
    }

};
#endif

