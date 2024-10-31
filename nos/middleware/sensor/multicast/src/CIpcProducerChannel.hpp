// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
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
#include "CPeerValidator.hpp"

using namespace std;
using namespace nvsipl;

class CIpcProducerChannel : public CChannel
{
  public:
    CIpcProducerChannel() = delete;
    CIpcProducerChannel(const string &name,
                        NvSciBufModule &bufMod,
                        NvSciSyncModule &syncMod,
                        SensorInfo *pSensorInfo,
                        CAppConfig *pAppConfig,
                        INvSIPLCamera *pCamera)
        : CChannel(name, bufMod, syncMod, pSensorInfo, pAppConfig)
        , m_pCamera(pCamera)
    {
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
        for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
            if (m_srcIpcHandles[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_srcIpcHandles[i]);
            }
            if (m_srcIpcEndpoints[i]) {
                (void)NvSciIpcCloseEndpointSafe(m_srcIpcEndpoints[i], false);
            }
        }
    }

    virtual SIPLStatus Init(void)
    {
        if (m_pAppConfig && m_pAppConfig->IsLateAttachEnabled()) {
            lateConsumerHelper = std::make_shared<CLateConsumerHelper>(m_bufModule, m_syncModule, m_pAppConfig);
            m_uEarlyConsCount = NUM_IPC_CONSUMERS - lateConsumerHelper->GetLateConsCount();
        }
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Deinit(void)
    {
        if (m_upProducer != nullptr) {
            return m_upProducer->Deinit();
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Post(NvSIPLBuffers &siplBuffers)
    {
        PLOG_DBG("Post\n");

        auto status = m_upProducer->Post(&siplBuffers);
        PCHK_STATUS_AND_RETURN(status, "Post");

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        CFactory &factory = CFactory::GetInstance(m_pAppConfig);

        m_upPoolManager = factory.CreatePoolManager(m_pSensorInfo->id, MAX_NUM_PACKETS);
        PCHK_PTR_AND_RETURN(m_upPoolManager, "factory.CreatePoolManager");
        PLOG_DBG("PoolManager is created.\n");
        m_upPoolManager->PreInit(lateConsumerHelper);

        CProducer *pProducer =
            factory.CreateProducer(ProducerType_SIPL, m_upPoolManager->GetHandle(), m_pSensorInfo->id);
        PCHK_PTR_AND_RETURN(pProducer, "factory, create SIPL producer.");
        CSIPLProducer *pSIPLProducer = dynamic_cast<CSIPLProducer *>(pProducer);
        pSIPLProducer->PreInit(m_pCamera, lateConsumerHelper);

        m_upProducer.reset(pProducer);
        PLOG_DBG("SIPL Producer is created.\n");

        m_upProducer->SetProfiler(pProfiler);
        PLOG_DBG("Producer is created.\n");

        if (NUM_IPC_CONSUMERS > 1) {
            auto status = factory.CreateMulticastBlock(NUM_IPC_CONSUMERS, m_multicastHandle);
            PCHK_STATUS_AND_RETURN(status, "factory.CreateMulticastBlock");
            PLOG_DBG("Multicast block is created.\n");
        }

        for (auto i = 0U; i < m_uEarlyConsCount; i++) {
            auto status = CreateIpcSrcAndEndpoint(GetSrcChannel(m_pSensorInfo->id, i), &m_srcIpcEndpoints[i],
                                                  &m_srcIpcHandles[i]);
            PCHK_STATUS_AND_RETURN(status, "Create ipc src block");
            PLOG_DBG("Ipc src block: %u is created, srcChannel: %s.\n", i, GetSrcChannel(m_pSensorInfo->id, i).c_str());
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void)
    {
        NvSciStreamEventType event;
        NvSciStreamBlock producerLink = 0U;
        NvSciStreamBlock consumerLink = 0U;
        NvSciError sciErr = NvSciError_Success;

        PLOG_DBG("Connect.\n");

        if (NUM_IPC_CONSUMERS == 1U) {
            producerLink = m_upProducer->GetHandle();
        } else {
            // connect producer with multicast
            sciErr = NvSciStreamBlockConnect(m_upProducer->GetHandle(), m_multicastHandle);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect producer to multicast");
            PLOG_DBG("Producer is connected to multicast.\n");
            producerLink = m_multicastHandle;
        }
        for (auto i = 0U; i < m_uEarlyConsCount; i++) {
            vector<NvSciStreamBlock> vConsumerLinks{};
            PopulateConsumerLinks(i, vConsumerLinks);
            if (vConsumerLinks.size() > 0) {
                consumerLink = vConsumerLinks[0];
                for (auto j = 1U; j < vConsumerLinks.size(); j++) {
                    sciErr = NvSciStreamBlockConnect(vConsumerLinks[j], consumerLink);
                    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "connect consumer link");
                    consumerLink = vConsumerLinks[j];
                }

                sciErr = NvSciStreamBlockConnect(producerLink, consumerLink);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "producerLink connects to consumerLink");
                PLOG_DBG("producerLink is connected to consumerLink: %u\n", i);
            }
        }

        // indicate Multicast to proceed with the initialization and streaming with the connected consumers
        if (m_multicastHandle && lateConsumerHelper && lateConsumerHelper->GetLateConsCount()) {
            sciErr = NvSciStreamBlockSetupStatusSet(m_multicastHandle, NvSciStreamSetup_Connect, true);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast status set to NvSciStreamSetup_Connect");
        }

        LOG_MSG("Producer is connecting to the stream...\n");
        vector<NvSciStreamBlock> vQueryBlocks{};
        PopulateQueryBlocks(vQueryBlocks);
        if (vQueryBlocks.size() > 0) {
            for (auto i = 0U; i < vQueryBlocks.size(); i++) {
                sciErr = NvSciStreamBlockEventQuery(vQueryBlocks[i], QUERY_TIMEOUT_FOREVER, &event);
                PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "query block");
                PLOG_DBG("Query block: %u is connected.\n", i);
            }
        }

        LOG_MSG("Producer is connected to the stream!\n");
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Reconcile(void)
    {
        SIPLStatus status = CChannel::Reconcile();

        NvSciStreamEventType event;
        // query multicast
        if (m_multicastHandle != 0U) {
            NvSciError sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT_FOREVER, &event);
            if (NvSciError_Success != sciErr || event != NvSciStreamEventType_SetupComplete) {
                PLOG_ERR("Multicast block setup error: 0x%.8X .\n", sciErr);
                return NVSIPL_STATUS_ERROR;
            }
            PLOG_DBG("Multicast block is setup complete.\n");
            m_isReadyForLateAttach = true;
        }

        return status;
    }

    SIPLStatus attach(void)
    {
        if (m_isLateConsumerAttached) {
            PLOG_WARN("Late consumer is already attached!");
            return NVSIPL_STATUS_OK;
        }

        if (!m_isReadyForLateAttach) {
            PLOG_WARN("It is NOT ready for attach now!");
            return NVSIPL_STATUS_OK;
        }
        m_isReadyForLateAttach = false;

        NvSciStreamEventType event;
        NvSciError sciErr = NvSciError_Success;
        PLOG_DBG("ConnectLateConsumer, make sure multicast status is NvSciStreamEventType_SetupComplete before attach "
                 "late consumer\n");

        // create ipcsrc block for late consumer
        uint32_t lateIdx = m_uEarlyConsCount;
        for (auto i = lateIdx; i < NUM_IPC_CONSUMERS; i++) {
            auto status =
                CreateIpcSrcAndEndpoint(GetSrcChannel(m_pSensorInfo->id, i), &m_lateIpcEndpoint, &m_srcIpcHandles[i]);
            PCHK_STATUS_AND_RETURN(status, "CFactory::Create ipc src Block");
        }

        // connect late consumers to multicast
        for (auto i = lateIdx; i < NUM_IPC_CONSUMERS; i++) {
            sciErr = NvSciStreamBlockConnect(m_multicastHandle, m_srcIpcHandles[i]);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to ipc src");
            PLOG_DBG("Multicast is connected to ipc src: %u\n", i);
        }

        // indicate multicast block to proceed with the initialization and streaming with the connectting consumers
        sciErr = NvSciStreamBlockSetupStatusSet(m_multicastHandle, NvSciStreamSetup_Connect, true);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast set to NvSciStreamSetup_Connect!");

        // make sure relevant blocks reach streaming phase.
        // query consumers and queues
        for (auto i = lateIdx; i < NUM_IPC_CONSUMERS; i++) {
            sciErr = NvSciStreamBlockEventQuery(m_srcIpcHandles[i], QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Ipc src");
            PLOG_DBG("Ipc src: %u is connected.\n", i);
        }

        // make sure late consumers attach success
        sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT, &event);
        if (NvSciError_Success != sciErr || event != NvSciStreamEventType_SetupComplete) {
            PLOG_ERR("Multicast block setup error: 0x%.8X .\n", sciErr);

            // release relevant resource if attach fail
            CFactory::GetInstance(m_pAppConfig).ReleaseIpcBlock(m_lateIpcEndpoint, m_srcIpcHandles[lateIdx]);
            m_srcIpcHandles[lateIdx] = 0U;
            m_lateIpcEndpoint = 0U;

            // reset this flag to allow next attach
            m_isReadyForLateAttach = true;
            return NVSIPL_STATUS_ERROR;
        }
        m_isReadyForLateAttach = true;
        m_isLateConsumerAttached = true;

        LOG_MSG("Late consumer is attached successfully!\n");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus detach(void)
    {
        if (!m_isLateConsumerAttached) {
            PLOG_WARN("Late consumer is already detached!");
            return NVSIPL_STATUS_OK;
        }

        uint32_t lateIdx = m_uEarlyConsCount;
        NvSciError sciErr = NvSciStreamBlockDisconnect(m_srcIpcHandles[lateIdx]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockDisconnect fail");
        m_isLateConsumerAttached = false;

        CFactory::GetInstance(m_pAppConfig).ReleaseIpcBlock(m_lateIpcEndpoint, m_srcIpcHandles[lateIdx]);
        m_srcIpcHandles[lateIdx] = 0U;
        m_lateIpcEndpoint = 0U;

        LOG_MSG("Late consumer is detached successfully!\n");
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
    virtual const string GetSrcChannel(uint32_t sensorId, uint32_t consumerId) const = 0;
    virtual SIPLStatus
    CreateIpcSrcAndEndpoint(const string &srcChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcSrc) = 0;

    virtual void PopulateConsumerLinks(uint32_t consumerId, vector<NvSciStreamBlock> &vConsumerLinks)
    {
        vConsumerLinks.push_back(m_srcIpcHandles[consumerId]);
    };

    virtual void PopulateQueryBlocks(vector<NvSciStreamBlock> &vQueryBlocks)
    {
        vQueryBlocks.push_back(m_upProducer->GetHandle());
        vQueryBlocks.push_back(m_upPoolManager->GetHandle());
        for (auto i = 0U; i < m_uEarlyConsCount; i++) {
            vQueryBlocks.push_back(m_srcIpcHandles[i]);
        }
        if (m_multicastHandle != 0U) {
            vQueryBlocks.push_back(m_multicastHandle);
        }
    };

    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler *> &vEventHandlers)
    {
        if (!isStreamRunning) {
            vEventHandlers.push_back(m_upPoolManager.get());
        }
        vEventHandlers.push_back(m_upProducer.get());
    }

    std::unique_ptr<CProducer> m_upProducer = nullptr;

  private:
    INvSIPLCamera *m_pCamera = nullptr;
    unique_ptr<CPoolManager> m_upPoolManager = nullptr;
    NvSciStreamBlock m_multicastHandle = 0U;
    NvSciStreamBlock m_srcIpcHandles[NUM_IPC_CONSUMERS]{};
    NvSciIpcEndpoint m_srcIpcEndpoints[NUM_IPC_CONSUMERS]{};

    bool m_isLateConsumerAttached = false;
    bool m_isReadyForLateAttach = false;
    uint32_t m_uEarlyConsCount = NUM_IPC_CONSUMERS;
    NvSciIpcEndpoint m_lateIpcEndpoint = 0U;
    std::shared_ptr<CLateConsumerHelper> lateConsumerHelper = nullptr;
};

class CP2pProducerChannel : public CIpcProducerChannel
{
  public:
    CP2pProducerChannel() = delete;
    CP2pProducerChannel(NvSciBufModule &bufMod,
                        NvSciSyncModule &syncMod,
                        SensorInfo *pSensorInfo,
                        CAppConfig *pAppConfig,
                        INvSIPLCamera *pCamera)
        : CIpcProducerChannel("P2PProdChan", bufMod, syncMod, pSensorInfo, pAppConfig, pCamera)
    {
    }

    virtual SIPLStatus Connect(void)
    {
        SIPLStatus status;
        if (!m_bFinishPeerValidation) {
            m_pPeerValidator.reset(new CPeerValidator(m_pAppConfig));
            CHK_PTR_AND_RETURN(m_pPeerValidator, "CPeerValidator creation");
            m_pPeerValidator->SetHandle(m_upProducer->GetHandle());
            status = m_pPeerValidator->SendValidationInfo();
            m_bFinishPeerValidation = true;
            PCHK_STATUS_AND_RETURN(status, "SendValidationInfo");
        }

        status = CIpcProducerChannel::Connect();
        PCHK_STATUS_AND_RETURN(status, "Connect");

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual const string GetSrcChannel(uint32_t sensorId, uint32_t consumerId) const
    {
        return IPC_CHANNEL_PREFIX + std::to_string(sensorId * NUM_IPC_CONSUMERS * 2 + 2 * consumerId + 0);
    }

    virtual SIPLStatus
    CreateIpcSrcAndEndpoint(const string &srcChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcSrc)
    {
        CFactory &factory = CFactory::GetInstance(m_pAppConfig);
        auto status = factory.CreateIpcBlock(m_syncModule, m_bufModule, srcChannel.c_str(), true, pEndPoint, pIpcSrc);
        PCHK_STATUS_AND_RETURN(status, "CFactory create ipc src block");

        return NVSIPL_STATUS_OK;
    }

    std::unique_ptr<CPeerValidator> m_pPeerValidator{ nullptr };
    static bool m_bFinishPeerValidation;
};
bool CP2pProducerChannel::m_bFinishPeerValidation{ false };

class CC2cProducerChannel : public CIpcProducerChannel
{
  public:
    CC2cProducerChannel() = delete;
    CC2cProducerChannel(NvSciBufModule &bufMod,
                        NvSciSyncModule &syncMod,
                        SensorInfo *pSensorInfo,
                        CAppConfig *pAppConfig,
                        INvSIPLCamera *pCamera)
        : CIpcProducerChannel("C2CProdChan", bufMod, syncMod, pSensorInfo, pAppConfig, pCamera)
    {
    }

    ~CC2cProducerChannel(void)
    {
        PLOG_DBG("Release\n");
        for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
            if (m_vQueueHandles[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_vQueueHandles[i]);
            }
            if (m_presentSyncs[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_presentSyncs[i]);
            }
        }
    }

    virtual SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        auto status = CIpcProducerChannel::CreateBlocks(pProfiler);
        PCHK_STATUS_AND_RETURN(status, "CIpcProducerChannel::CreateBlocks");

        if (m_pAppConfig->GetQueueType() != QueueType_Mailbox) {
            return NVSIPL_STATUS_OK;
        }

        for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
            CFactory &factory = CFactory::GetInstance(m_pAppConfig);
            status = factory.CreatePresentSync(m_syncModule, m_presentSyncs[i]);
            PCHK_STATUS_AND_RETURN(status, "factory.CreatePresentSync");
        }

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual const string GetSrcChannel(uint32_t sensorId, uint32_t consumerId) const
    {
        return C2C_SRC_CHANNEL_PREFIX + std::to_string(sensorId * NUM_IPC_CONSUMERS + consumerId + 1);
    }

    virtual SIPLStatus
    CreateIpcSrcAndEndpoint(const string &srcChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcSrc)
    {
        NvSciStreamBlock queueHandle = 0U;

        CFactory &factory = CFactory::GetInstance(m_pAppConfig);

        auto status = factory.CreateQueue(m_pAppConfig->GetQueueType(), &queueHandle);
        PCHK_STATUS_AND_RETURN(status, "CFactory create c2c queue handle");
        m_vQueueHandles.push_back(queueHandle);

        status = factory.CreateC2CSrc(m_syncModule, m_bufModule, srcChannel.c_str(), queueHandle, pEndPoint, pIpcSrc);
        PCHK_STATUS_AND_RETURN(status, "factory.CreateC2CSrc");

        return NVSIPL_STATUS_OK;
    }

    virtual void PopulateConsumerLinks(uint32_t consumerId, vector<NvSciStreamBlock> &vConsumerLinks)
    {
        CIpcProducerChannel::PopulateConsumerLinks(consumerId, vConsumerLinks);
        if (m_presentSyncs[consumerId] != 0U) {
            vConsumerLinks.push_back(m_presentSyncs[consumerId]);
        }
    }

    virtual void PopulateQueryBlocks(vector<NvSciStreamBlock> &vQueryBlocks)
    {
        CIpcProducerChannel::PopulateQueryBlocks(vQueryBlocks);
        for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
            vQueryBlocks.push_back(m_vQueueHandles[i]);
            if (m_presentSyncs[i] != 0U) {
                vQueryBlocks.push_back(m_presentSyncs[i]);
            }
        }
    };

  private:
    vector<NvSciStreamBlock> m_vQueueHandles;
    NvSciStreamBlock m_presentSyncs[NUM_IPC_CONSUMERS]{};
};

#endif
