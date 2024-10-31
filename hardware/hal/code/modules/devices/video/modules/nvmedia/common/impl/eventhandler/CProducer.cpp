// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <algorithm>
#include "CProducer.hpp"

CProducer::CProducer(std::string name, NvSciStreamBlock handle, uint32_t uSensor, std::shared_ptr<CAttributeProvider> attrProvider):
    CClientCommon(name, handle, uSensor)
{
   m_numBuffersWithConsumer = 0U;
   m_attrProvider = attrProvider;
}

SIPLStatus CProducer::HandleStreamInit(void)
{
    /* Query number of consumers */
    auto sciErr = NvSciStreamBlockConsumerCountGet(m_handle, &m_numConsumers);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer query number of consumers");
    if (m_numConsumers > NUM_CONSUMERS) {
        PLOG_ERR("%s:Consumer count is too big: %u\r\n", m_name.c_str(), m_numConsumers);
        return NVSIPL_STATUS_ERROR;
    }
    m_numWaitSyncObj = m_numConsumers;
    // If enable lateAttach, the comsumer count queried it the value passed into multicast block when created,
    // including late consumer, but the producer only need to handle the early consumer's sync obj.
    // Here the late consumer is hard code to 1, so the m_numWaitSyncObj reduce by 1.
    if(m_enableLateAttach) {
        assert(m_numWaitSyncObj >= 2);
        --m_numWaitSyncObj;
    }

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
            PLOG_ERR("%s:Didn't receive expected PacketReady event.\r\n", m_name.c_str());
            return NVSIPL_STATUS_ERROR;
        }
        sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketGet");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CProducer::HandleSyncExport(void)
{
    auto sciErr = NvSciError_Success;

    std::vector<uint32_t> processedIds;

    for (int i = 0; i < static_cast<int>(m_elemsInfo.size()); ++i)
    {
        printf("CProducer::HandleSyncExport m_elemsInfo userType: %d, isUsed: %d, hasSibling: %d\n", 
               (int)m_elemsInfo[i].userType, 
               m_elemsInfo[i].isUsed, 
               m_elemsInfo[i].hasSibling);
    }
    for (uint32_t i = 0U; i < m_elemsInfo.size(); ++i) {
        if (m_elemsInfo[i].userType == ELEMENT_TYPE_METADATA || !m_elemsInfo[i].isUsed ||
            processedIds.end() != std::find(processedIds.begin(), processedIds.end(), i)) {
            continue;
        }
        // Merge and reconcile sync attrs.
        std::vector<NvSciSyncAttrList> unreconciled;
        auto status = CollectWaiterAttrList(i, unreconciled);
        PCHK_STATUS_AND_RETURN(status, "CollectWaiterAttrList");

        // If it has slbling, collect the waiter attribute list one by one.
        // For example, there is only one shared sync object for ISP0&ISP1 buffer.
        if (m_elemsInfo[i].hasSibling) {
            for (uint32_t j = i + 1; j < m_elemsInfo.size(); ++j) {
                if (!m_elemsInfo[j].hasSibling) {
                    continue;
                }

                auto status = CollectWaiterAttrList(j, unreconciled);
                PCHK_STATUS_AND_RETURN(status, "CollectWaiterAttrList");
                processedIds.push_back(j);
            }
        }

        if (unreconciled.empty()) {
            continue;
        }

        uint32_t waiterNum = unreconciled.size();
        unreconciled.push_back(m_signalerAttrLists[i]);
        if (m_cpuWaitAttr) {
            unreconciled.push_back(m_cpuWaitAttr);
        }

        NvSciSyncAttrList cudaWaiterAttrList = nullptr;
        if (m_enableLateAttach) {
            assert(m_attrProvider != nullptr);

            SIPLStatus status = NVSIPL_STATUS_OK;
            status = m_attrProvider->GetSyncWaiterAttrList(ConsumerType::CUDA_CONSUMER, &cudaWaiterAttrList);
            CHK_STATUS_AND_RETURN(status, "m_attrProvider->GetBufAttrList.");
            unreconciled.push_back(cudaWaiterAttrList);
        }

        NvSciSyncAttrList reconciled = nullptr;
        NvSciSyncAttrList conflicts = nullptr;

        sciErr = NvSciSyncAttrListReconcile(unreconciled.data(), unreconciled.size(), &reconciled, &conflicts);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");

        if (m_enableLateAttach)
        {
            NvSciSyncAttrListFree(unreconciled.back());
            unreconciled.pop_back();
        }

        for (uint32_t m = 0U; m < waiterNum; ++m) {
            NvSciSyncAttrListFree(unreconciled[m]);
        }

    /* Allocate sync object */
        NvSciSyncObj &signalSyncObj = m_signalSyncObjs[i];
        sciErr = NvSciSyncObjAlloc(reconciled, &signalSyncObj);
        NvSciSyncAttrListFree(reconciled);
        NvSciSyncAttrListFree(conflicts);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");

        status = RegisterSignalSyncObj(m_elemsInfo[i].userType, signalSyncObj);
        PCHK_STATUS_AND_RETURN(status, "RegisterSignalSyncObj");

        status = SetSignalObject(i, signalSyncObj);
        PCHK_STATUS_AND_RETURN(status, "SetSignalObject");

        // If it has sibling, set the same sync object.
        if (m_elemsInfo[i].hasSibling) {
            for (uint32_t k = i + 1; k < m_elemsInfo.size(); ++k) {
                if (!m_elemsInfo[k].hasSibling) {
                    continue;
                }

                status = SetSignalObject(k, signalSyncObj);
                PCHK_STATUS_AND_RETURN(status, "SetSignalObject");
            }
        }
    }
    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr import");

    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj export");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CProducer::HandlePayload(void)
{
    NvSciStreamCookie cookie;
    uint32_t packetIndex = 0;

    if (m_numBuffersWithConsumer == 0U) {
        PLOG_WARN("%s:HandlePayload, m_numBuffersWithConsumer is 0\r\n", m_name.c_str());
        return NVSIPL_STATUS_OK;
    }
    PLOG_DBG("%s:HandlePayload, m_numBuffersWithConsumer: %u\r\n", m_name.c_str(), m_numBuffersWithConsumer.load());
    auto sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Obtain packet for payload");

    m_numBuffersWithConsumer--;
    auto status = GetIndexFromCookie(cookie, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "GetIndexFromCookie");

    ClientPacket *packet = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(packet, "Get packet by cookie\r\n");

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

            PLOG_DBG("%s:Query fence from consumer: %d, m_interested element id = %d\n", m_name.c_str(), i, j);

            NvSciSyncFence prefence = NvSciSyncFenceInitializer;
            sciErr = NvSciStreamBlockPacketFenceGet(m_handle, packet->handle, i, j, &prefence);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("%s:Failed (0x%x) to query fence from consumer: %d\r\n", m_name.c_str(), sciErr, i);
                return NVSIPL_STATUS_ERROR;
            }

            uint64_t id;
            uint64_t value;
            sciErr = NvSciSyncFenceExtractFence(&prefence, &id, &value);
            if (NvSciError_ClearedFence == sciErr) {
                PLOG_DBG("%s:Empty fence supplied as prefence.Skipping prefence insertion \r\n", m_name.c_str());
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
    PLOG_DBG("%s:Post, m_numBuffersWithConsumer: %u\r\n", m_name.c_str(), m_numBuffersWithConsumer.load());

    if (m_pProfiler != nullptr) {
        m_pProfiler->OnFrameAvailable();
    }

    return NVSIPL_STATUS_OK;
}

NvSciBufAttrValAccessPerm CProducer::GetMetaPerm(void)
{
    return NvSciBufAccessPerm_ReadWrite;
}
