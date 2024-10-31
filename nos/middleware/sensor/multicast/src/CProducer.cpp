// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CProducer.hpp"

CProducer::CProducer(std::string name, NvSciStreamBlock handle, uint32_t uSensor)
    : CClientCommon(name, handle, uSensor)
{
    m_numBuffersWithConsumer = 0U;
}

SIPLStatus CProducer::HandleStreamInit(void)
{
    /* Query number of consumers */
    auto sciErr = NvSciStreamBlockConsumerCountGet(m_handle, &m_numConsumers);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer query number of consumers");

    LOG_MSG("Producer: Consumer count is %u\n", m_numConsumers);

    if (m_numConsumers > MAX_NUM_CONSUMERS) {
        PLOG_ERR("Consumer count is too big: %u\n", m_numConsumers);
        return NVSIPL_STATUS_ERROR;
    }
    m_numWaitSyncObj = m_numConsumers;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CProducer::HandleSetupComplete(void)
{
    CClientCommon::HandleSetupComplete();

    NvSciStreamEventType eventType;
    NvSciStreamCookie cookie;

    // Producer receives notification and takes initial ownership of packets
    for (uint32_t i = 0U; i < m_numPacket; i++) {
        NvSciError sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &eventType);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Get initial ownership of packet");

        if (eventType != NvSciStreamEventType_PacketReady) {
            PLOG_ERR("Didn't receive expected PacketReady event.\n");
            return NVSIPL_STATUS_ERROR;
        }
        sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketGet");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CProducer::HandlePayload(void)
{
    NvSciStreamCookie cookie;
    uint32_t packetIndex = 0;

    if (m_numBuffersWithConsumer == 0U) {
        PLOG_WARN("HandlePayload, m_numBuffersWithConsumer is 0\n");
        return NVSIPL_STATUS_OK;
    }
    PLOG_DBG("HandlePayload, m_numBuffersWithConsumer: %u\n", m_numBuffersWithConsumer.load());

    auto sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Obtain packet for payload");

    m_numBuffersWithConsumer--;
    auto status = GetIndexFromCookie(cookie, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "GetIndexFromCookie");

    ClientPacket *packet = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(packet, "Get packet by cookie\n");

    /* Query fences for this element from each consumer */
    for (uint32_t i = 0U; i < m_numWaitSyncObj; ++i) {
        for (uint32_t j = 0U; j < m_elemsInfo.size(); ++j) {
            /* If the received waiter obj if NULL,
             * the consumer is done using this element,
             * skip waiting on pre-fence.
             */
            if (nullptr == m_waiterSyncObjs[i][j]) {
                continue;
            }

            PLOG_DBG("Query fence from consumer: %d, m_interested element id = %d\n", i, j);

            NvSciSyncFence prefence = NvSciSyncFenceInitializer;
            sciErr = NvSciStreamBlockPacketFenceGet(m_handle, packet->handle, i, j, &prefence);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed (0x%x) to query fence from consumer: %d\n", sciErr, i);
                return NVSIPL_STATUS_ERROR;
            }

            uint64_t id;
            uint64_t value;
            sciErr = NvSciSyncFenceExtractFence(&prefence, &id, &value);
            if (NvSciError_ClearedFence == sciErr) {
                PLOG_DBG("Empty fence supplied as prefence.Skipping prefence insertion \n");
                continue;
            }

            // Perform CPU wait to WAR the issue of failing to register sync object with ISP.
            if (m_cpuWaitContext != nullptr) {
                sciErr = NvSciSyncFenceWait(&prefence, m_cpuWaitContext, FENCE_FRAME_TIMEOUT_US);
                NvSciSyncFenceClear(&prefence);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait prefence");
            } else {
                status = InsertPrefence(packetIndex, prefence);
                NvSciSyncFenceClear(&prefence);
                PCHK_STATUS_AND_RETURN(status, "Insert prefence");
            }
        }
    }

    OnPacketGotten(packetIndex);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CProducer::Post(void *pBuffer)
{
    uint32_t packetIndex = 0;

    auto status = MapPayload(pBuffer, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "MapPayload");

    NvSciSyncFence postFence = NvSciSyncFenceInitializer;

    status = GetPostfence(packetIndex, &postFence);
    PCHK_STATUS_AND_RETURN(status, "GetPostFence");

    /* Update postfence for this element */
    auto sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[packetIndex].handle, 0U, &postFence);

    sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[packetIndex].handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent");

    NvSciSyncFenceClear(&postFence);

    m_numBuffersWithConsumer++;
    PLOG_DBG("Post, m_numBuffersWithConsumer: %u\n", m_numBuffersWithConsumer.load());

    if (m_pProfiler != nullptr) {
        m_pProfiler->OnFrameAvailable();
    }

    return NVSIPL_STATUS_OK;
}

NvSciBufAttrValAccessPerm CProducer::GetMetaPerm(void)
{
    return NvSciBufAccessPerm_ReadWrite;
}
