// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CSINGLEPROCESSCHANNEL
#define CSINGLEPROCESSCHANNEL

#include "CChannel.hpp"
#include "CFactory.hpp"
#include "CPoolManager.hpp"
#include "CClientCommon.hpp"

using namespace std;
using namespace nvsipl;

class CSingleProcessChannel: public CChannel
{
public:
    CSingleProcessChannel() = delete;
    CSingleProcessChannel(NvSciBufModule& bufMod,
        NvSciSyncModule& syncMod, SensorInfo *pSensorInfo, INvSIPLCamera* pCamera) :
        CChannel("SingleProcChan", bufMod, syncMod, pSensorInfo)
    {
        m_pCamera = pCamera;
    }

    ~CSingleProcessChannel(void)
    {
        PLOG_DBG("Release.\n");

        if (m_upPoolManager != nullptr && m_upPoolManager->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_upPoolManager->GetHandle());
        }
        if (m_multicastHandle != 0U) {
            (void)NvSciStreamBlockDelete(m_multicastHandle);
        }
        if (m_vClients[0] != nullptr && m_vClients[0]->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_vClients[0]->GetHandle());
        }
        for (uint32_t i = 1U; i < m_vClients.size(); i++) {
            CConsumer* pConsumer = dynamic_cast<CConsumer*>(m_vClients[i].get());
            if (pConsumer != nullptr && pConsumer->GetHandle() != 0U) {
                (void)NvSciStreamBlockDelete(pConsumer->GetHandle());
            }
            if (pConsumer != nullptr && pConsumer->GetQueueHandle() != 0U) {
                (void)NvSciStreamBlockDelete(pConsumer->GetQueueHandle());
            }
        }
    }
    SIPLStatus Post(INvSIPLClient::INvSIPLBuffer *pBuffer)
    {
        PLOG_DBG("Post\n");

        CProducer* pProducer = dynamic_cast<CProducer*>(m_vClients[0].get());
        PCHK_PTR_AND_RETURN(pProducer, "m_vClients[0] converts to CProducer");
        auto status = pProducer->Post(pBuffer);
        PCHK_STATUS_AND_RETURN(status, "Post");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        m_upPoolManager = CFactory::CreatePoolManager(m_pSensorInfo->id);
        CHK_PTR_AND_RETURN(m_upPoolManager, "CFactory::CreatePoolManager.");
        PLOG_DBG("PoolManager is created.\n");

        std::unique_ptr<CProducer> upProducer = CFactory::CreateProducer(m_upPoolManager->GetHandle(), m_pSensorInfo->id, m_pCamera);
        PCHK_PTR_AND_RETURN(upProducer, "CFactory::CreateProducer.");
        PLOG_DBG("Producer is created.\n");

        upProducer->SetProfiler(pProfiler);
        m_vClients.push_back(std::move(upProducer));

        std::unique_ptr<CConsumer> upCUDAConsumer = CFactory::CreateConsumer(CUDA_CONSUMER, m_pSensorInfo, m_bUseMailbox);
        PCHK_PTR_AND_RETURN(upCUDAConsumer, "CFactory::Create CUDA consumer");
        m_vClients.push_back(std::move(upCUDAConsumer));
        PLOG_DBG("CUDA consumer is created.\n");

        SIPLStatus status;
        if (MAX_NUM_CONSUMERS > 1U) {
            status = CFactory::CreateMulticastBlock(MAX_NUM_CONSUMERS, m_multicastHandle);
            PCHK_STATUS_AND_RETURN(status, "CFactory::CreateMulticastBlock");
            PLOG_DBG("Multicast block is created.\n");

            std::unique_ptr<CConsumer> upEncConsumer = CFactory::CreateConsumer(ENC_CONSUMER, m_pSensorInfo, m_bUseMailbox);
            PCHK_PTR_AND_RETURN(upEncConsumer, "CFactory::Create encoder consumer");
            m_vClients.push_back(std::move(upEncConsumer));
            PLOG_DBG("Encoder consumer is created.\n");
        }
        if (MAX_NUM_CONSUMERS > 2U) {
            std::unique_ptr<CConsumer> upCommonConsumer = CFactory::CreateConsumer(COMMON_CONSUMER, m_pSensorInfo, m_bUseMailbox);
            PCHK_PTR_AND_RETURN(upCommonConsumer, "CFactory::Create common consumer");
            m_vClients.push_back(std::move(upCommonConsumer));
            PLOG_DBG("Common consumer is created.\n");
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void)
    {
        NvSciStreamEventType event;

        PLOG_DBG("Connect.\n");

        if (MAX_NUM_CONSUMERS == 1U) {
            auto sciErr = NvSciStreamBlockConnect(m_vClients[0]->GetHandle(), m_vClients[1]->GetHandle());
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, ("Producer connect to" + m_vClients[1]->GetName()).c_str());
        } else {
            //connect producer with multicast
            auto sciErr = NvSciStreamBlockConnect(m_vClients[0]->GetHandle(), m_multicastHandle);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect producer to multicast");
            PLOG_DBG("Producer is connected to multicast.\n");

            //connect multicast with each consumer
            for (uint32_t i = 1U; i < m_vClients.size(); i++) {
                sciErr = NvSciStreamBlockConnect(m_multicastHandle, m_vClients[i]->GetHandle());
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to consumer");
                PLOG_DBG("Multicast is connected to consumer: %u\n", (i-1));
            }
        }

        LOG_MSG("Connecting to the stream...\n");
        //query producer
        auto sciErr = NvSciStreamBlockEventQuery(m_vClients[0]->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "producer");
        PLOG_DBG("Producer is connected.\n");

        sciErr = NvSciStreamBlockEventQuery(m_upPoolManager->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "pool");
        PLOG_DBG("Pool is connected.\n");

        //query consumers and queues
        for (uint32_t i = 1U; i < m_vClients.size(); i++) {
            CConsumer* pConsumer = dynamic_cast<CConsumer*>(m_vClients[i].get());
		    sciErr = NvSciStreamBlockEventQuery(pConsumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "queue");
            PLOG_DBG("Queue:%u is connected.\n", (i-1));

            sciErr = NvSciStreamBlockEventQuery(pConsumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "consumer");
            PLOG_DBG("Consumer:%u is connected.\n", (i-1));
        }

        //query multicast
        if (m_multicastHandle != 0U) {
            sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "multicast");
            PLOG_DBG("Multicast is connected.\n");
        }

        LOG_MSG("All blocks are connected to the stream!\n");
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus InitBlocks(void)
    {
        PLOG_DBG("InitBlocks.\n");

        auto status = m_upPoolManager->Init();
        PCHK_STATUS_AND_RETURN(status, "Pool Init");

        for (auto& upClient: m_vClients) {
            auto status = upClient->Init(m_bufModule, m_syncModule);
            PCHK_STATUS_AND_RETURN(status, (upClient->GetName() + " Init").c_str());
        }

        return NVSIPL_STATUS_OK;
    }

    virtual void SetConsumerConfig(const ConsumerConfig& consConfig)
    {
        for (uint32_t i = 1U; i < m_vClients.size(); i++) {
            CConsumer* pConsumer = dynamic_cast<CConsumer*>(m_vClients[i].get());
            if (pConsumer != nullptr) {
                pConsumer->SetConsumerConfig(consConfig);
            }
        }
        m_bUseMailbox = consConfig.bUseMailbox;
    }

protected:
    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler*>& vEventHandlers)
    {
        if (!isStreamRunning) {
            vEventHandlers.push_back(m_upPoolManager.get());
        }
        for (auto& upClient: m_vClients) {
            vEventHandlers.push_back(upClient.get());
        }
    }

private:

    INvSIPLCamera *m_pCamera = nullptr;
    unique_ptr<CPoolManager> m_upPoolManager = nullptr;
    NvSciStreamBlock m_multicastHandle = 0U;
    vector<unique_ptr<CClientCommon>> m_vClients;
    bool m_bUseMailbox = false;
};

#endif
