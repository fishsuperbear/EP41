// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CConsumer.hpp"

CConsumer::CConsumer(std::string name, NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle, ConsumerType i_consumertype) :
    CClientCommon(name, handle, uSensor)
{
    m_queueHandle = queueHandle;
    m_consConfig.bFileDump = false;
    m_consConfig.frameMod = 1U;
    _consumertype = i_consumertype;
}

SIPLStatus CConsumer::HandlePayload(void)
{
    NvSciStreamCookie cookie;
    uint32_t packetIndex = 0;

    /* Obtain packet with the new payload */
    auto sciErr = NvSciStreamConsumerPacketAcquire(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketAcquire");
    PLOG_DBG("%s:Acquired a packet (cookie = %u).\r\n", m_name.c_str(), cookie);

    auto status = GetIndexFromCookie(cookie, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "GetIndexFromCookie");

    ClientPacket *packet = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(packet, "GetPacketByCookie");

    if (m_pProfiler != nullptr) {
        m_pProfiler->OnFrameAvailable();
    }

    m_frameNum++;
    if (m_frameNum % m_consConfig.frameMod != 0) {
        /* Release the packet back to the producer */
        sciErr = NvSciStreamConsumerPacketRelease(m_handle, packet->handle);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");
        return NVSIPL_STATUS_OK;
    }

    uint32_t elementId = 0U;

    /* If the received waiter obj is NULL,
     * the producer is done writing data into this element, skip waiting on pre-fence.
    * For consumer, there is only one waiter, using index 0 as default.
     */
    for (; elementId < m_elemsInfo.size(); ++elementId) {
        if (m_waiterSyncObjs[0U][elementId] != nullptr) {

            PLOG_DBG("Get prefence and insert it, waiter sync objects = %x\n", m_waiterSyncObjs[0U][elementId]);

            NvSciSyncFence prefence = NvSciSyncFenceInitializer;
            /* Query fences for this element from producer */
            sciErr = NvSciStreamBlockPacketFenceGet(m_handle, packet->handle, 0U, elementId, &prefence);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceGet");

            status = InsertPrefence(packetIndex, prefence);
            NvSciSyncFenceClear(&prefence);
            PCHK_STATUS_AND_RETURN(status, ": InsertPrefence");

            break;
        }
    }

    status = SetEofSyncObj();
    PCHK_STATUS_AND_RETURN(status, "SetEofSyncObj");

    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    status = ProcessPayload(packetIndex, &postfence);
    PCHK_STATUS_AND_RETURN(status, "ProcessPayload");

    if (m_cpuWaitContext != nullptr) {
        sciErr = NvSciSyncFenceWait(&postfence, m_cpuWaitContext, FENCE_FRAME_TIMEOUT_US);
        NvSciSyncFenceClear(&postfence);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");
    } else {
        sciErr = NvSciStreamBlockPacketFenceSet(m_handle, packet->handle, elementId, &postfence);
        NvSciSyncFenceClear(&postfence);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceSet");
    }

    status = OnProcessPayloadDone(packetIndex);
    PCHK_STATUS_AND_RETURN(status, "OnProcessPayloadDone");

    /* Release the packet back to the producer */
    sciErr = NvSciStreamConsumerPacketRelease(m_handle, packet->handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");

    return NVSIPL_STATUS_OK;
}

ConsumerType CConsumer::GetConsumerType()
{
    return _consumertype;
}

NvSciStreamBlock CConsumer::GetQueueHandle(void)
{
    return m_queueHandle;
}

NvSciBufAttrValAccessPerm CConsumer::GetMetaPerm(void)
{
    return NvSciBufAccessPerm_Readonly;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CConsumer::MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    PLOG_DBG("%s:Mapping meta buffer, packetIndex: %u.\n", m_name.c_str(), packetIndex);
    auto sciErr = NvSciBufObjGetConstCpuPtr(bufObj, (void const **)&m_metaPtrs[packetIndex]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetConstCpuPtr");

    return NVSIPL_STATUS_OK;
}

void CConsumer::SetConsumerConfig(const ConsumerConfig& consConfig)
{
   m_consConfig = consConfig;
}

SIPLStatus CConsumer::SetUnusedElement(uint32_t elementId)
{
    auto err = NvSciStreamBlockElementUsageSet(m_handle, elementId, false);
    PCHK_NVSCISTATUS_AND_RETURN(err, "NvSciStreamBlockElementUsageSet");

    return NVSIPL_STATUS_OK;
}
