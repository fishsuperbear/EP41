// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "nvplayfair.h"
#include "CCustProducer.hpp"

constexpr static int32_t OUTPUT_TYPE_UNDEFINED = -1;

CCustProducer::CCustProducer(std::string name, NvSciStreamBlock handle, uint32_t uSensor)
    : CClientCommon(name, handle, uSensor)
{
    m_numBuffersWithConsumer = 0U;
}

SIPLStatus CCustProducer::HandleStreamInit(void)
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

SIPLStatus CCustProducer::HandleSetupComplete(void)
{
    /* NvSciStreamEventType eventType; */
    /* NvSciStreamCookie cookie; */

    /* // Producer receives notification and takes initial ownership of packets */
    /* for (uint32_t i = 0U; i < m_numPacket; i++) { */
    /*     NvSciError sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &eventType); */
    /*     printf("CCustProducer::HandleSetupComplete ++ sciErr = %d\n",sciErr); */
    /*     PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Get initial ownership of packet"); */

    /*     if (eventType != NvSciStreamEventType_PacketReady) { */
    /*         PLOG_ERR("Didn't receive expected PacketReady event.\n"); */
    /*         return NVSIPL_STATUS_ERROR; */
    /*     } */
    /*     HandlePayload(); */
    /*     /1* printf("++++++++++++++++++CCustProducer::HandlePayload\n"); *1/ */
    /*     /1* sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie); *1/ */
    /*     /1* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketGet"); *1/ */
    /*     /1* uint32_t packetIndex = 0; *1/ */
    /*     /1* auto status = GetIndexFromCookie(cookie, packetIndex); *1/ */
    /*     /1* insertToPacketIndexs(packetIndex); *1/ */
    /* } */
    /* printf("CCustProducer::HandleSetupComplete\n"); */

    return NVSIPL_STATUS_OK;
}

constexpr int64_t  FENCE_WAIT_INFINITE{ -1 };
NvpRateLimitInfo_t              rateLimitInfo;
SIPLStatus CCustProducer::HandlePayload(void)
{
    printf("++++++++++++++++++CCustProducer::HandlePayload\n");
#if 1
    // Get a packet
    NvSciStreamCookie cookie{ 0U };
    CHECK_NVSCIERR(NvSciStreamProducerPacketGet(m_handle,
                                                &cookie));
    ClientPacket *packet = GetPacketByCookie(cookie);

    if(m_numBuffersWithConsumer==0){
        NvpRateLimitInit(&rateLimitInfo,30);
        NvpMarkPeriodicExecStart(&rateLimitInfo);
    }else{
        NvpRateLimitWait(&rateLimitInfo);
    }

    NvSciSyncFence prefence;
    NvSciStreamBlockPacketFenceGet(m_handle,
                                                          packet->handle,
                                                          0,
                                                          0,
                                                          &prefence);
    NvSciSyncFenceWait(&prefence,
                                              m_cpuWaitContext,
                                              FENCE_WAIT_INFINITE);
    NvSciSyncFenceClear(&prefence);

    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    NvSciSyncObjGenerateFence(m_signalSyncObjs[0],&postfence);
    NvSciStreamBlockPacketFenceSet(m_handle,
                                                          packet->handle,
                                                          0,
                                                          &postfence);

    NvSciStreamProducerPacketPresent(m_handle,
                                                    packet->handle);
    NvSciSyncObjSignal(m_signalSyncObjs[0]);
    NvSciSyncFenceClear(&postfence);
    m_numBuffersWithConsumer++;
    uint32_t packetIndex = 0;
    auto sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Obtain packet for payload");
    auto status = GetIndexFromCookie(cookie, packetIndex);
    insertToPacketIndexs(packetIndex);
    return NVSIPL_STATUS_OK;
#else
    NvSciStreamCookie cookie;
    uint32_t packetIndex = 0;

    /* if (m_numBuffersWithConsumer == 0U) { */
    /*     PLOG_WARN("HandlePayload, m_numBuffersWithConsumer is 0\n"); */
    /*     return NVSIPL_STATUS_OK; */
    /* } */
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
        for (uint32_t j = 0U; j < 1; ++j) {
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

    memset(m_metaPtrs[packetIndex],0x5E,1024);
    NvSciSyncFence postFence = NvSciSyncFenceInitializer;
    NvSciSyncObjGenerateFence(m_signalSyncObjs[0],&postFence);
    sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[packetIndex].handle, 0U, &postFence);

    sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[packetIndex].handle);
    printf("get NvSciStreamProducerPacketPresent=%d\n",sciErr);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent");

    NvSciSyncObjSignal(m_signalSyncObjs[0]);
    NvSciSyncFenceClear(&postFence);
    /* insertToPacketIndexs(packetIndex); */

    return NVSIPL_STATUS_OK;
#endif
}

SIPLStatus CCustProducer::MapPayload(void *pBuffer, uint32_t &packetIndex)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CCustProducer::post(uint32_t pIndex)
{
    /* printf("pIndex = %d\n", pIndex); */
    /* NvSciSyncFence postFence = NvSciSyncFenceInitializer; */
    /* /1* NvSciSyncObjGenerateFence(m_signalSyncObjs[0],&postFence); *1/ */
    /* auto sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[pIndex].handle, 0U, &postFence); */

    /* sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[pIndex].handle); */
    /* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent"); */

    /* /1* NvSciSyncObjSignal(m_signalSyncObjs[0]); *1/ */
    /* NvSciSyncFenceClear(&postFence); */

    /* m_numBuffersWithConsumer++; */
    /* printf("Post, m_numBuffersWithConsumer: %u\n", m_numBuffersWithConsumer.load()); */

    /* // if (m_pProfiler != nullptr) { */
    /* //     m_pProfiler->OnFrameAvailable(); */
    /* // } */

    return NVSIPL_STATUS_OK;
}
/* SIPLStatus CCustProducer::Post(void *pBuffer) */
/* { */
/*     uint32_t packetIndex = 0; */

/*     /1* auto status = MapPayload(pBuffer, packetIndex); *1/ */
/*     /1* PCHK_STATUS_AND_RETURN(status, "MapPayload"); *1/ */

/*     NvSciSyncFence postFence = NvSciSyncFenceInitializer; */

/*     auto status = GetPostfence(packetIndex, &postFence); */
/*     PCHK_STATUS_AND_RETURN(status, "GetPostFence"); */

/*     /1* Update postfence for this element *1/ */
/*     auto sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[packetIndex].handle, 0U, &postFence); */

/*     sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[packetIndex].handle); */
/*     PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent"); */

/*     NvSciSyncFenceClear(&postFence); */

/*     m_numBuffersWithConsumer++; */
/*     PLOG_DBG("Post, m_numBuffersWithConsumer: %u\n", m_numBuffersWithConsumer.load()); */

/*     // if (m_pProfiler != nullptr) { */
/*     //     m_pProfiler->OnFrameAvailable(); */
/*     // } */

/*     return NVSIPL_STATUS_OK; */
/* } */

NvSciBufAttrValAccessPerm CCustProducer::GetMetaPerm(void)
{
    return NvSciBufAccessPerm_ReadWrite;
}

void CCustProducer::OnPacketGotten(uint32_t packetIndex)
{
    /* for (uint32_t i = 0U; i < MAX_OUTPUTS_PER_SENSOR; ++i) { */
    /*     if (!m_siplBuffers[i].nvmBuffers.empty() && m_siplBuffers[i].nvmBuffers[packetIndex] != nullptr) { */
    /*         m_siplBuffers[i].nvmBuffers[packetIndex]->Release(); */
    /*         LOG_DBG("CCustProducer::OnPacketGotten, ISP type %u packet gotten \n", i); */
    /*     } */
    /* } */
}

SIPLStatus CCustProducer::HandleClientInit(void)
{
    /* m_elemTypeToOutputType[ELEMENT_TYPE_NV12_BL] = static_cast<int32_t>(INvSIPLClient::ConsumerDesc::OutputType::ISP0); */
    /* m_elemTypeToOutputType[ELEMENT_TYPE_NV12_PL] = static_cast<int32_t>(INvSIPLClient::ConsumerDesc::OutputType::ISP1); */
    /* m_elemTypeToOutputType[ELEMENT_TYPE_ICP_RAW] = static_cast<int32_t>(INvSIPLClient::ConsumerDesc::OutputType::ICP); */

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CCustProducer::MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    /* PLOG_DBG("Mapping meta buffer, packetIndex: %u.\n", packetIndex); */
    /* auto sciErr = NvSciBufObjGetCpuPtr(bufObj, (void **)&m_metaPtrs[packetIndex]); */
    /* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetCpuPtr"); */

    return NVSIPL_STATUS_OK;
}

void CCustProducer::setEndpointBufAttr(NvSciBufAttrList attrList)
{
    NvSciBufType bufType{ NvSciBufType_RawBuffer };
    // Convert buffer size from MB to Bytes
    uint64_t rawsize{ static_cast<uint64_t>(1024U * 1024U * 6) };
    uint64_t align{ 4 * 1024 };
    NvSciBufAttrValAccessPerm perm{ NvSciBufAccessPerm_ReadWrite };
    // Disable cpu access for vidmem
    bool cpuaccess_flag = true;

    NvSciBufAttrKeyValuePair rawbuffattrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
        { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
    };

    CHECK_NVSCIERR(NvSciBufAttrListSetAttrs(attrList,
                    rawbuffattrs,
                    sizeof(rawbuffattrs) / sizeof(NvSciBufAttrKeyValuePair)));

}
