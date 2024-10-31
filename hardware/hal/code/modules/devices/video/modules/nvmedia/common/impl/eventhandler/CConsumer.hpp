/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CCONSUMER_HPP
#define CCONSUMER_HPP

#include "hw_nvmedia_eventhandler_common_impl.h"

#include "nvscibuf.h"
#include "CClientCommon.hpp"
#include <atomic>

class CConsumer: public CClientCommon
{
public:
    /** @brief Default constructor. */
    CConsumer() = delete;
    CConsumer(std::string name, NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle, ConsumerType i_consumertype);
    /** @brief Default destructor. */
    virtual ~CConsumer() = default;
public:
    ConsumerType GetConsumerType();
    // Streaming functions
    NvSciStreamBlock GetQueueHandle(void);
    void SetConsumerConfig(const ConsumerConfig& consConfig);
public:
    virtual hw_ret_s32 RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
        HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig) = 0;

protected:
    SIPLStatus HandlePayload(void) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) = 0;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) = 0;
    virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus SetUnusedElement(uint32_t elementId) override;

    uint32_t m_frameNum = 0U;
    ConsumerConfig m_consConfig;

private:
    NvSciStreamBlock m_queueHandle = 0U;
    /* Virtual address for the meta buffer */
    ConsumerType    _consumertype;
};
#endif
