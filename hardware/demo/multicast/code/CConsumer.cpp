// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CConsumer.hpp"

CConsumer::CConsumer(std::string name, NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle) :
    CClientCommon(name, handle, uSensor)
{
    m_queueHandle = queueHandle;
    m_consConfig.bFileDump = false;
    m_consConfig.frameMod = 1U;
}

SIPLStatus CConsumer::HandlePayload(void)
{
    NvSciStreamCookie cookie;
    uint32_t packetIndex = 0;

    /* Obtain packet with the new payload */
    auto sciErr = NvSciStreamConsumerPacketAcquire(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketAcquire");
    PLOG_DBG("Acquired a packet (cookie = %u).\n", cookie);

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

    /* If the received waiter obj if NULL,
     * the producer is done writing data into this element, skip waiting on pre-fence.
     */
    if (nullptr != m_waiterSyncObjs[0]) {
        NvSciSyncFence prefence = NvSciSyncFenceInitializer;
        /* Query fences for this element from producer */
        sciErr = NvSciStreamBlockPacketFenceGet(m_handle, packet->handle, 0U, m_dataIndex, &prefence);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceGet");

        status = InsertPrefence(packetIndex, prefence);
        PCHK_STATUS_AND_RETURN(status, ": InsertPrefence");

        NvSciSyncFenceClear(&prefence);
    }

    status = SetEofSyncObj();
    PCHK_STATUS_AND_RETURN(status, "SetEofSyncObj");

    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    status = ProcessPayload(packetIndex, &postfence);
    PCHK_STATUS_AND_RETURN(status, "ProcessPayload");

    if (m_cpuWaitContext != nullptr) {
        sciErr = NvSciSyncFenceWait(&postfence, m_cpuWaitContext, FENCE_FRAME_TIMEOUT_US);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");
    }

    status = OnProcessPayloadDone(packetIndex);
    PCHK_STATUS_AND_RETURN(status, "OnProcessPayloadDone");

    /* Release the packet back to the producer */
    sciErr = NvSciStreamConsumerPacketRelease(m_handle, packet->handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");

    NvSciSyncFenceClear(&postfence);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceSet");

    return NVSIPL_STATUS_OK;
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
SIPLStatus CConsumer::MapMetaBuffer(uint32_t packetIndex)
{
    PLOG_DBG("Mapping meta buffer, packetIndex: %u.\n", packetIndex);
    auto sciErr = NvSciBufObjGetConstCpuPtr(m_packets[packetIndex].metaObj, (void const**)&m_metaPtrs[packetIndex]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetConstCpuPtr");

    return NVSIPL_STATUS_OK;
}

void CConsumer::SetConsumerConfig(const ConsumerConfig& consConfig)
{
   m_consConfig = consConfig;
}
