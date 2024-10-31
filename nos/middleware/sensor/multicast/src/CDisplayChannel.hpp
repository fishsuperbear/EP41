// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CDISPLAY_CHANNEL_HPP
#define CDISPLAY_CHANNEL_HPP

#include "CChannel.hpp"
#include "CFactory.hpp"
#include "CPoolManager.hpp"
#include "CClientCommon.hpp"

using namespace nvsipl;

class CDisplayChannel : public CChannel
{
  private:
    struct PipelineInfo
    {
        std::shared_ptr<CDisplayProducer> prod = nullptr;
        std::vector<std::unique_ptr<CConsumer>> cons;
        std::unique_ptr<CPoolManager> pool = nullptr;
        NvSciStreamBlock multicastHandle = 0U;
    };

  public:
    CDisplayChannel() = delete;
    CDisplayChannel(NvSciBufModule &bufMod,
                    NvSciSyncModule &syncMod,
                    SensorInfo *pSensorInfo,
                    CAppConfig *pAppConfig,
                    const std::shared_ptr<COpenWFDController> &wfdController)
        : CChannel("CDisplayChannel", bufMod, syncMod, pSensorInfo, pAppConfig)
    {
        m_spWFDController = wfdController;
    }

    virtual ~CDisplayChannel(void)
    {
        if (m_pipeline.pool != nullptr && m_pipeline.pool->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_pipeline.pool->GetHandle());
        }

        if (m_pipeline.prod != nullptr && m_pipeline.prod->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_pipeline.prod->GetHandle());
            m_pipeline.prod.reset();
        }

        for (auto &consumer : m_pipeline.cons) {
            if (consumer != nullptr && consumer->GetHandle() != 0U) {
                (void)NvSciStreamBlockDelete(consumer->GetHandle());
            }

            if (consumer != nullptr && consumer->GetQueueHandle() != 0U) {
                (void)NvSciStreamBlockDelete(consumer->GetQueueHandle());
            }

            consumer.reset();
        }

        if (m_pipeline.multicastHandle != 0U) {
            (void)NvSciStreamBlockDelete(m_pipeline.multicastHandle);
        }

        PLOG_DBG("CDisplayChannel released end\n");
    }

    SIPLStatus Deinit(void)
    {
        if (m_pipeline.prod != nullptr) {
            m_pipeline.prod->Deinit();
        }

        for (auto &consumer : m_pipeline.cons) {
            consumer->Deinit();
        }

        return NVSIPL_STATUS_OK;
    }

    std::shared_ptr<CDisplayProducer> &GetDisplayProducer()
    {
        return m_pipeline.prod;
    }

    SIPLStatus CreatePipeline(PipelineInfo &info, CProfiler *pProfiler)
    {
        uint32_t numPackets = 3U;

        CFactory &factory = CFactory::GetInstance(m_pAppConfig);

        info.pool = factory.CreatePoolManager(m_pSensorInfo->id, numPackets);
        CHK_PTR_AND_RETURN(info.pool, "factory.CreatePoolManager.");
        PLOG_DBG("PoolManager is created.\n");

        LOG_MSG("CDisplayChannel: Sensor id is: %d \n", m_pSensorInfo->id);

        std::unique_ptr<CConsumer> upDisplayConsumer =
            factory.CreateConsumer(ConsumerType_Display, m_pSensorInfo->id, QueueType_Mailbox);
        PCHK_PTR_AND_RETURN(upDisplayConsumer, "factory, create display consumer");

        CDisplayConsumer *pDisplayConsumer = dynamic_cast<CDisplayConsumer *>(upDisplayConsumer.get());
        pDisplayConsumer->PreInit(m_spWFDController);

        info.cons.push_back(std::move(upDisplayConsumer));
        PLOG_DBG("Display consumer is created.\n");

        CProducer *pProducer = factory.CreateProducer(ProducerType_Display, info.pool->GetHandle());
        PCHK_PTR_AND_RETURN(pProducer, "factory, create display producer.");
        CDisplayProducer *pDisplayProducer = dynamic_cast<CDisplayProducer *>(pProducer);
        pDisplayProducer->PreInit(numPackets);

        info.prod.reset(pDisplayProducer);
        PLOG_DBG("Display Producer is created.\n");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        auto status = CreatePipeline(m_pipeline, pProfiler);
        PCHK_STATUS_AND_RETURN(status, "CreatePipeline");

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void)
    {
        PLOG_DBG("Connect.\n");

        NvSciStreamEventType event;

        auto sciErr = NvSciError_Success;
        if (0 == m_pipeline.multicastHandle) {
            sciErr = NvSciStreamBlockConnect(m_pipeline.prod->GetHandle(), m_pipeline.cons[0]->GetHandle());
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, ("Producer connect to" + m_pipeline.cons[0]->GetName()).c_str());
        } else {
            //connect producer with multicast
            auto sciErr = NvSciStreamBlockConnect(m_pipeline.prod->GetHandle(), m_pipeline.multicastHandle);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect producer to multicast");
            PLOG_DBG("Producer is connected to multicast.\n");

            //connect multicast with each consumer
            for (const auto &consumer : m_pipeline.cons) {
                sciErr = NvSciStreamBlockConnect(m_pipeline.multicastHandle, consumer->GetHandle());
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to consumer");
                PLOG_DBG("Multicast is connected to consumer: %s\n", consumer->GetName().c_str());
            }
        }

        sciErr = NvSciStreamBlockEventQuery(m_pipeline.prod->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "producer");
        PLOG_DBG("Producer is connected.\n");

        sciErr = NvSciStreamBlockEventQuery(m_pipeline.pool->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "pool");
        PLOG_DBG("Pool connected.\n");

        for (const auto &consumer : m_pipeline.cons) {
            sciErr = NvSciStreamBlockEventQuery(consumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "queue");
            PLOG_DBG("Queue for %s is connected.\n", consumer->GetName().c_str());

            sciErr = NvSciStreamBlockEventQuery(consumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "consumer");
            PLOG_DBG("Consumer:%s is connected.\n", consumer->GetName().c_str());
        }

        if (m_pipeline.multicastHandle != 0U) {
            sciErr = NvSciStreamBlockEventQuery(m_pipeline.multicastHandle, QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "multicast");
            PLOG_DBG("Multicast is connected.\n");
        }

        LOG_MSG("All blocks are connected to the stream!\n");
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus InitBlocks(void)
    {
        PLOG_DBG("InitBlocks.\n");

        auto status = m_pipeline.pool->Init();
        PCHK_STATUS_AND_RETURN(status, "Pool Init");

        status = m_pipeline.prod->Init(m_bufModule, m_syncModule);
        PCHK_STATUS_AND_RETURN(status, (m_pipeline.prod->GetName() + " Init").c_str());

        for (const auto &consumer : m_pipeline.cons) {
            status = consumer->Init(m_bufModule, m_syncModule);
            PCHK_STATUS_AND_RETURN(status, (consumer->GetName() + " Init").c_str());
        }

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler *> &vEventHandlers)
    {
        if (!isStreamRunning) {
            vEventHandlers.push_back(m_pipeline.pool.get());
        }

        vEventHandlers.push_back(m_pipeline.prod.get());
        for (const auto &consumer : m_pipeline.cons) {
            vEventHandlers.push_back(consumer.get());
        }
    }

  private:
    INvSIPLCamera *m_pCamera = nullptr;

    std::shared_ptr<COpenWFDController> m_spWFDController = nullptr;

    PipelineInfo m_pipeline;
};

#endif
