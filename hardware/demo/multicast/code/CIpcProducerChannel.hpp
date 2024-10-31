// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CIPCPRODUCERCHANNEL
#define CIPCPRODUCERCHANNEL

#include "CChannel.hpp"
#include "CFactory.hpp"
#include "CPoolManager.hpp"

using namespace std;
using namespace nvsipl;

class CIpcProducerChannel: public CChannel
{
public:
    CIpcProducerChannel() = delete;
    CIpcProducerChannel(NvSciBufModule& bufMod,
        NvSciSyncModule& syncMod, SensorInfo *pSensorInfo, INvSIPLCamera* pCamera) :
        CChannel("IpcProdChan", bufMod, syncMod, pSensorInfo)
    {
        m_pCamera = pCamera;
        for (auto i = 0U; i < MAX_NUM_CONSUMERS; i++) {
            m_srcChannels[i] = "nvscistream_" + std::to_string(pSensorInfo->id * MAX_NUM_CONSUMERS * 2 + 2*i + 0);
            m_srcIpcHandles[i] = 0U;
        }
    }

    ~CIpcProducerChannel(void)
    {
        PLOG_DBG("Release.\n");

        if (m_upPoolManager != nullptr && m_upPoolManager->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_upPoolManager->GetHandle());
        }
        if (m_multicastHandle != 0U) {
            (void)NvSciStreamBlockDelete(m_multicastHandle);
        }
        if (m_upProducer != nullptr && m_upProducer->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_upProducer->GetHandle());
        }
        for (auto i = 0U; i < MAX_NUM_CONSUMERS; i++) {
            if (m_srcIpcHandles[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_srcIpcHandles[i]);
            }
        }
    }

    SIPLStatus Post(INvSIPLClient::INvSIPLBuffer *pBuffer)
    {
        PLOG_DBG("Post\n");

        auto status = m_upProducer->Post(pBuffer);
        PCHK_STATUS_AND_RETURN(status, "Post");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        m_upPoolManager = CFactory::CreatePoolManager(m_pSensorInfo->id);
        PCHK_PTR_AND_RETURN(m_upPoolManager, "CFactory::CreatePoolManager");
        PLOG_DBG("PoolManager is created.\n");

        m_upProducer = CFactory::CreateProducer(m_upPoolManager->GetHandle(), m_pSensorInfo->id, m_pCamera);
        PCHK_PTR_AND_RETURN(m_upProducer, "CFactory::CreateProducer");
        m_upProducer->SetProfiler(pProfiler);
        PLOG_DBG("Producer is created.\n");

        if (MAX_NUM_CONSUMERS > 1) {
            auto status = CFactory::CreateMulticastBlock(MAX_NUM_CONSUMERS, m_multicastHandle);
            PCHK_STATUS_AND_RETURN(status, "CFactory::CreateMulticastBlock");
            PLOG_DBG("Multicast block is created.\n");
        }

        for (auto i = 0U; i < MAX_NUM_CONSUMERS; i++) {
            auto status = CFactory::CreateIpcBlock(m_syncModule, m_bufModule, m_srcChannels[i].c_str(), true, &m_srcIpcHandles[i]);
            PCHK_STATUS_AND_RETURN(status, "CFactory::Create ipc src Block");
            PLOG_DBG("Ipc src block: %u is created.\n", i);
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void)
    {
        NvSciStreamEventType event;

        PLOG_DBG("Connect.\n");

        if (MAX_NUM_CONSUMERS == 1U){
            auto sciErr = NvSciStreamBlockConnect(m_upProducer->GetHandle(), m_srcIpcHandles[0]);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer connect to ipc src");
            PLOG_DBG("Producer is connected to ipc src.\n");
        } else {
            //connect producer with multicast
            auto sciErr = NvSciStreamBlockConnect(m_upProducer->GetHandle(), m_multicastHandle);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect producer to multicast");
            PLOG_DBG("Producer is connected to multicast.\n");

            for (auto i = 0U; i < MAX_NUM_CONSUMERS; i++) {
                sciErr = NvSciStreamBlockConnect(m_multicastHandle, m_srcIpcHandles[i]);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to ipc src");
                PLOG_DBG("Multicast is connected to ipc src: %u\n", i);
            }
        }

        LOG_MSG("Producer is connecting to the stream...\n");
        //query producer
        PLOG_DBG("Query producer connection.\n");
        auto sciErr = NvSciStreamBlockEventQuery(m_upProducer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Producer");
        PLOG_DBG("Producer is connected.\n");

        PLOG_DBG("Query pool connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_upPoolManager->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Pool");
        PLOG_DBG("Pool is connected.\n");

        PLOG_DBG("Query ipc src connection.\n");
        //query consumers and queues
        for (auto i = 0U; i < MAX_NUM_CONSUMERS; i++) {
            sciErr = NvSciStreamBlockEventQuery(m_srcIpcHandles[i], QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Ipc src");
            PLOG_DBG("Ipc src: %u is connected.\n", i);
        }

        //query multicast
        if (m_multicastHandle != 0U) {
            PLOG_DBG("Query multicast block.\n");
            sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Multicast");
            PLOG_DBG("Multicast block is connected.\n");
        }

        LOG_MSG("Producer is connected to the stream!\n");
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus InitBlocks(void)
    {
        PLOG_DBG(": InitBlocks.\n");

        auto status = m_upPoolManager->Init();
        PCHK_STATUS_AND_RETURN(status, "Pool Init");

        status = m_upProducer->Init(m_bufModule, m_syncModule);
        PCHK_STATUS_AND_RETURN(status, (m_upProducer->GetName() + " Init").c_str());

        return NVSIPL_STATUS_OK;
    }

protected:
    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler*>& vEventHandlers)
    {
        if (!isStreamRunning) {
            vEventHandlers.push_back(m_upPoolManager.get());
        }
        vEventHandlers.push_back(m_upProducer.get());
    }

private:

    INvSIPLCamera *m_pCamera = nullptr;
    unique_ptr<CPoolManager> m_upPoolManager = nullptr;
    NvSciStreamBlock m_multicastHandle = 0U;
    std::unique_ptr<CProducer> m_upProducer = nullptr;
    NvSciStreamBlock m_srcIpcHandles[MAX_NUM_CONSUMERS];
    string m_srcChannels[MAX_NUM_CONSUMERS];
};

#endif
