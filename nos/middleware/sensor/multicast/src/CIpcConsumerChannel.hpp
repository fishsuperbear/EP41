// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.
#ifndef CIPCCONSUMERCHANNEL
#define CIPCCONSUMERCHANNEL

#include "CChannel.hpp"
#include "CFactory.hpp"
#include "CClientCommon.hpp"
#include "CPeerValidator.hpp"

using namespace std;
using namespace nvsipl;

class CIpcConsumerChannel : public CChannel
{
  public:
    CIpcConsumerChannel() = delete;
    CIpcConsumerChannel(const string &name,
                        NvSciBufModule &bufMod,
                        NvSciSyncModule &syncMod,
                        SensorInfo *pSensorInfo,
                        CAppConfig *pAppConfig)
        : CChannel(name, bufMod, syncMod, pSensorInfo, pAppConfig)
    {
    }

    ~CIpcConsumerChannel(void)
    {
        PLOG_DBG("Release.\n");

        if (m_upConsumer != nullptr) {
            if (m_upConsumer->GetQueueHandle() != 0U) {
                (void)NvSciStreamBlockDelete(m_upConsumer->GetQueueHandle());
            }
            if (m_upConsumer->GetHandle() != 0U) {
                (void)NvSciStreamBlockDelete(m_upConsumer->GetHandle());
            }
        }

        if (m_dstIpcHandle != 0U) {
            (void)NvSciStreamBlockDelete(m_dstIpcHandle);
        }

        if (m_dstIpcEndpoint) {
            (void)NvSciIpcCloseEndpointSafe(m_dstIpcEndpoint, false);
        }
    }

    virtual SIPLStatus Deinit(void)
    {
        if (m_upConsumer != nullptr) {
            return m_upConsumer->Deinit();
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        CFactory &factory = CFactory::GetInstance(m_pAppConfig);
        m_upConsumer = factory.CreateConsumer(m_pAppConfig->GetConsumerType(), m_pSensorInfo->id);
        PCHK_PTR_AND_RETURN(m_upConsumer, "factory.CreateConsumer");

        m_upConsumer->SetProfiler(pProfiler);
        PLOG_DBG((m_upConsumer->GetName() + " is created.\n").c_str());

        auto status = CreateIpcDstAndEndpoint(GetDstChannel(), &m_dstIpcEndpoint, &m_dstIpcHandle);
        PCHK_STATUS_AND_RETURN(status, "Create ipc dst block");
        PLOG_DBG("Dst ipc block is created, dstChannel: %s.\n", GetDstChannel().c_str());

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void)
    {
        NvSciStreamEventType event;

        PLOG_DBG("Connect.\n");

        auto sciErr = NvSciStreamBlockConnect(m_dstIpcHandle, m_upConsumer->GetHandle());
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect blocks: dstIpc - consumer");

        LOG_MSG((m_upConsumer->GetName() + " is connecting to the stream...\n").c_str());
        LOG_DBG("Query ipc dst connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_dstIpcHandle, QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "ipc dst");
        PLOG_DBG("Ipc dst is connected.\n");

        //query consumer and queue
        PLOG_DBG("Query queue connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_upConsumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "queue");
        PLOG_DBG("Queue is connected.\n");

        PLOG_DBG("Query consumer connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_upConsumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "consumer");
        PLOG_DBG("Consumer is connected.\n");
        LOG_MSG((m_upConsumer->GetName() + " is connected to the stream!\n").c_str());

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus InitBlocks(void)
    {
        PLOG_DBG("InitBlocks.\n");

        auto status = m_upConsumer->Init(m_bufModule, m_syncModule);
        PCHK_STATUS_AND_RETURN(status, (m_upConsumer->GetName() + " Init.").c_str());

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual const string GetDstChannel() const = 0;
    virtual SIPLStatus
    CreateIpcDstAndEndpoint(const string &dstChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcDst) = 0;

    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler *> &vEventHandlers)
    {
        vEventHandlers.push_back(m_upConsumer.get());
    }

    std::unique_ptr<CConsumer> m_upConsumer = nullptr;

  private:
    NvSciStreamBlock m_dstIpcHandle = 0U;
    NvSciIpcEndpoint m_dstIpcEndpoint = 0U;
};

class CP2pConsumerChannel : public CIpcConsumerChannel
{
  public:
    CP2pConsumerChannel() = delete;
    CP2pConsumerChannel(NvSciBufModule &bufMod,
                        NvSciSyncModule &syncMod,
                        SensorInfo *pSensorInfo,
                        CAppConfig *pAppConfig)
        : CIpcConsumerChannel("P2PConsChan", bufMod, syncMod, pSensorInfo, pAppConfig)
    {
    }

    virtual SIPLStatus Connect(void)
    {
        if (!m_bFinishPeerValidation) {
            m_pPeerValidator.reset(new CPeerValidator(m_pAppConfig));
            CHK_PTR_AND_RETURN(m_pPeerValidator, "CPeerValidator creation");
            m_pPeerValidator->SetHandle(m_upConsumer->GetHandle());
        }

        auto status = CIpcConsumerChannel::Connect();
        PCHK_STATUS_AND_RETURN(status, "Connect");

        if (!m_bFinishPeerValidation) {
            status = m_pPeerValidator->Validate();
            m_bFinishPeerValidation = true;
            PCHK_STATUS_AND_RETURN(status, "Validate peer info");
        }

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual const string GetDstChannel() const
    {
        return IPC_CHANNEL_PREFIX +
               std::to_string(m_pSensorInfo->id * NUM_IPC_CONSUMERS * 2 + 2 * m_pAppConfig->GetConsumerType() + 1);
    }

    virtual SIPLStatus
    CreateIpcDstAndEndpoint(const string &dstChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcDst)
    {
        CFactory &factory = CFactory::GetInstance(m_pAppConfig);
        auto status = factory.CreateIpcBlock(m_syncModule, m_bufModule, dstChannel.c_str(), false, pEndPoint, pIpcDst);
        PCHK_STATUS_AND_RETURN(status, "CFactory create ipc dst block");

        return NVSIPL_STATUS_OK;
    }

    std::unique_ptr<CPeerValidator> m_pPeerValidator{ nullptr };
    static bool m_bFinishPeerValidation;
};
bool CP2pConsumerChannel::m_bFinishPeerValidation{ false };

class CC2cConsumerChannel : public CIpcConsumerChannel
{
  public:
    CC2cConsumerChannel() = delete;
    CC2cConsumerChannel(NvSciBufModule &bufMod,
                        NvSciSyncModule &syncMod,
                        SensorInfo *pSensorInfo,
                        CAppConfig *pAppConfig)
        : CIpcConsumerChannel("C2CConsChan", bufMod, syncMod, pSensorInfo, pAppConfig)
    {
    }

    ~CC2cConsumerChannel(void)
    {
        if (m_upPoolManager != nullptr && m_upPoolManager->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_upPoolManager->GetHandle());
        }
    }

    virtual SIPLStatus Connect(void)
    {
        NvSciStreamEventType event;

        auto status = CIpcConsumerChannel::Connect();
        PCHK_STATUS_AND_RETURN(status, "CIpcConsumerChannel::Connect()");

        if (m_upPoolManager != nullptr && m_upPoolManager->GetHandle() != 0U) {
            auto sciErr = NvSciStreamBlockEventQuery(m_upPoolManager->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
            PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "c2c pool");
            PLOG_DBG("c2c pool is connected.\n");
        }

        return NVSIPL_STATUS_OK;
    }

  protected:
    virtual const string GetDstChannel() const
    {
        return C2C_DST_CHANNEL_PREFIX +
               std::to_string(m_pSensorInfo->id * NUM_IPC_CONSUMERS + m_pAppConfig->GetConsumerType() + 1);
    }

    virtual SIPLStatus
    CreateIpcDstAndEndpoint(const string &dstChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcDst)
    {
        vector<uint32_t> vElemTypesToSkip{};

        CFactory &factory = CFactory::GetInstance(m_pAppConfig);
        m_upPoolManager = factory.CreatePoolManager(m_pSensorInfo->id, MAX_NUM_PACKETS, true);
        PCHK_PTR_AND_RETURN(m_upPoolManager, "CFactory::CreatePoolManager");

        const vector<ElementInfo> &vElemInfo = m_upConsumer->GetPacketElementsInfo();
        for (ElementInfo elemInfo : vElemInfo) {
            if (!elemInfo.isUsed) {
                vElemTypesToSkip.push_back(elemInfo.userType);
            }
        }
        m_upPoolManager->SetElemTypesToSkip(vElemTypesToSkip);

        auto status = factory.CreateC2CDst(m_syncModule, m_bufModule, dstChannel.c_str(), m_upPoolManager->GetHandle(),
                                           pEndPoint, pIpcDst);
        PCHK_STATUS_AND_RETURN(status, "CFactory::CreateC2CDst");

        return NVSIPL_STATUS_OK;
    }

    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler *> &vEventHandlers)
    {
        CIpcConsumerChannel::GetEventThreadHandlers(isStreamRunning, vEventHandlers);

        if (!isStreamRunning && m_upPoolManager != nullptr) {
            vEventHandlers.push_back(m_upPoolManager.get());
        }
    }

  private:
    unique_ptr<CPoolManager> m_upPoolManager = nullptr;
};

#endif
