// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
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
#include "CDisplayProducer.hpp"

using namespace std;
using namespace nvsipl;

class CSingleProcessChannel : public CChannel
{
  public:
    CSingleProcessChannel() = delete;
    CSingleProcessChannel(NvSciBufModule &bufMod,
                          NvSciSyncModule &syncMod,
                          SensorInfo *pSensorInfo,
                          CAppConfig *pAppConfig,
                          INvSIPLCamera *pCamera,
                          std::shared_ptr<CDisplayProducer> spDisplayProducer)
        : CChannel("SingleProcChan", bufMod, syncMod, pSensorInfo, pAppConfig)
    {
        m_pCamera = pCamera;
        m_spDisplayProd = spDisplayProducer;
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
            m_vClients[0].reset();
        }
        for (uint32_t i = 1U; i < m_vClients.size(); i++) {
            CConsumer *pConsumer = dynamic_cast<CConsumer *>(m_vClients[i].get());
            if (pConsumer != nullptr && pConsumer->GetHandle() != 0U) {
                (void)NvSciStreamBlockDelete(pConsumer->GetHandle());
            }
            if (pConsumer != nullptr && pConsumer->GetQueueHandle() != 0U) {
                (void)NvSciStreamBlockDelete(pConsumer->GetQueueHandle());
            }
            m_vClients[i].reset();
        }
    }

    virtual SIPLStatus Deinit(void)
    {
        for (auto &upClient : m_vClients) {
            upClient->Deinit();
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Post(NvSIPLBuffers &siplBuffers)
    {
        PLOG_DBG("Post\n");

        CProducer *pProducer = dynamic_cast<CProducer *>(m_vClients[0].get());
        PCHK_PTR_AND_RETURN(pProducer, "m_vClients[0] converts to CProducer");

        auto status = pProducer->Post(&siplBuffers);
        PCHK_STATUS_AND_RETURN(status, "Post");

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        std::unique_ptr<CProducer> upProducer{ nullptr };

        PLOG_DBG("CreateBlocks.\n");

        CFactory &factory = CFactory::GetInstance(m_pAppConfig);

        m_upPoolManager = factory.CreatePoolManager(m_pSensorInfo->id, MAX_NUM_PACKETS);
        CHK_PTR_AND_RETURN(m_upPoolManager, "factory.CreatePoolManager.");
        PLOG_DBG("PoolManager is created.\n");

        CProducer *pProducer =
            factory.CreateProducer(ProducerType_SIPL, m_upPoolManager->GetHandle(), m_pSensorInfo->id);
        PCHK_PTR_AND_RETURN(pProducer, "factory, create SIPL producer.");

        CSIPLProducer *pSIPLProducer = dynamic_cast<CSIPLProducer *>(pProducer);
        pSIPLProducer->PreInit(m_pCamera);

        upProducer.reset(pProducer);
        PLOG_DBG("SIPL Producer is created.\n");

        upProducer->SetProfiler(pProfiler);
        m_vClients.push_back(std::move(upProducer));

        std::unique_ptr<CConsumer> upCUDAConsumer = factory.CreateConsumer(ConsumerType_Cuda, m_pSensorInfo->id);
        PCHK_PTR_AND_RETURN(upCUDAConsumer, "factory.Create CUDA consumer");

        m_vClients.push_back(std::move(upCUDAConsumer));
        PLOG_DBG("CUDA consumer is created.\n");

        if (m_spDisplayProd) {
            std::unique_ptr<CConsumer> upStitchingConsumer =
                factory.CreateConsumer(ConsumerType_Stitch, m_pSensorInfo->id);
            PCHK_PTR_AND_RETURN(upStitchingConsumer, "factory, Create stitching consumer");

            CStitchingConsumer *pStitchConsumer = dynamic_cast<CStitchingConsumer *>(upStitchingConsumer.get());
            pStitchConsumer->PreInit(m_spDisplayProd);

            m_vClients.push_back(std::move(upStitchingConsumer));
            PLOG_DBG("Stitching consumer is created.\n");
        }

        std::unique_ptr<CConsumer> upEncConsumer = factory.CreateConsumer(ConsumerType_Enc, m_pSensorInfo->id);
        PCHK_PTR_AND_RETURN(upEncConsumer, "factory, Create encoder consumer");

        m_vClients.push_back(std::move(upEncConsumer));
        PLOG_DBG("Encoder consumer is created.\n");

        auto status = factory.CreateMulticastBlock(m_vClients.size() - 1, m_multicastHandle);
        PCHK_STATUS_AND_RETURN(status, "factory.CreateMulticastBlock");
        PLOG_DBG("Multicast block is created.\n");

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
                PLOG_DBG("Multicast is connected to consumer: %u\n", (i - 1));
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
            CConsumer *pConsumer = dynamic_cast<CConsumer *>(m_vClients[i].get());
            sciErr = NvSciStreamBlockEventQuery(pConsumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "queue");
            PLOG_DBG("Queue:%u is connected.\n", (i - 1));

            sciErr = NvSciStreamBlockEventQuery(pConsumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "consumer");
            PLOG_DBG("Consumer:%u is connected.\n", (i - 1));
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

        for (auto &upClient : m_vClients) {
            auto status = upClient->Init(m_bufModule, m_syncModule);
            PCHK_STATUS_AND_RETURN(status, (upClient->GetName() + " Init").c_str());
        }

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler *> &vEventHandlers)
    {
        if (!isStreamRunning) {
            vEventHandlers.push_back(m_upPoolManager.get());
        }
        for (auto &upClient : m_vClients) {
            vEventHandlers.push_back(upClient.get());
        }
    }

  private:
    std::shared_ptr<CDisplayProducer> m_spDisplayProd = nullptr;
    INvSIPLCamera *m_pCamera = nullptr;
    unique_ptr<CPoolManager> m_upPoolManager = nullptr;
    NvSciStreamBlock m_multicastHandle = 0U;
    vector<unique_ptr<CClientCommon>> m_vClients;
};

#endif
