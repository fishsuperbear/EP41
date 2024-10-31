// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.
#ifndef CIPCCONSUMERCHANNEL
#define CIPCCONSUMERCHANNEL

#include "sensor/nvs_consumer/CChannel.hpp"
#include "sensor/nvs_consumer/CFactory.hpp"
#include "sensor/nvs_consumer/CClientCommon.hpp"
#include "sensor/nvs_consumer/COpenWFDController.hpp"


using namespace std;
using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace desay { 

#define COMSUMER_CHANNEL_NAME_PREFIX "nvscistream_camc_"

class CIpcConsumerChannel: public CChannel
{
public:
    CIpcConsumerChannel() = delete;
    CIpcConsumerChannel(NvSciBufModule& bufMod,
        NvSciSyncModule& syncMod, SensorInfo *pSensorInfo, ConsumerType consumerType,uint32_t channelStrIndex,uint32_t multicastIndex) :
        CChannel("P2PConsChan", bufMod, syncMod, pSensorInfo)
    {
        m_consumerType = consumerType;
        // m_dstChannel = COMSUMER_CHANNEL_NAME_PREFIX + std::to_string(channelStrIndex+multicastIndex);
        m_dstChannel = std::string("cam") + std::to_string(channelStrIndex) + "_recv" + std::to_string(multicastIndex);
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
        if (m_spWFDController) {
            m_spWFDController->DeInit();
        }

        if (m_upConsumer != nullptr) {
            return m_upConsumer->Deinit();
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        m_upConsumer = CFactory::CreateConsumer(m_consumerType, m_pSensorInfo, m_bUseMailbox);
        PCHK_PTR_AND_RETURN(m_upConsumer, "CFactory::CreateConsumer");
        // m_upConsumer->SetProfiler(pProfiler);
        PLOG_DBG((m_upConsumer->GetName() + " is created.\n").c_str());

        auto status = CreateIpcDstAndEndpoint(GetDstChannel(), &m_dstIpcEndpoint, &m_dstIpcHandle);
        PCHK_STATUS_AND_RETURN(status, "Create ipc dst block");
        PLOG_DBG("Dst ipc block is created, dstChannel: %s.\n", GetDstChannel().c_str());

        if (DISPLAY_CONSUMER == m_consumerType) {
            m_spWFDController = std::make_shared<COpenWFDController>();
            CHK_PTR_AND_RETURN(m_spWFDController, "COpenWFDController");

            status = m_spWFDController->InitResource();
            CHK_STATUS_AND_RETURN(status, " m_spWFDController->InitResource()");

            CDisplayConsumer *pDisplayConsumer = dynamic_cast<CDisplayConsumer *>(m_upConsumer.get());
            pDisplayConsumer->PreInit(m_spWFDController);
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void)
    {
        NvSciStreamEventType event;

        PLOG_DBG("Connect.\n");

        auto sciErr = NvSciStreamBlockConnect(m_dstIpcHandle, m_upConsumer->GetHandle());
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect blocks: dstIpc - consumer");

        LOG_MSG((m_upConsumer->GetName() + " is connecting to the stream["+GetDstChannel()+"]...\n").c_str());
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

    SIPLStatus ConnectWithTimeOut(int64_t microseconds)
    {
        NvSciStreamEventType event;

        PLOG_DBG("Connect.\n");

        auto sciErr = NvSciStreamBlockConnect(m_dstIpcHandle, m_upConsumer->GetHandle());
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect blocks: dstIpc - consumer");

        LOG_MSG((m_upConsumer->GetName() + " is connecting to the stream["+GetDstChannel()+"]...\n").c_str());
        LOG_DBG("Query ipc dst connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_dstIpcHandle, microseconds, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "ipc dst");
        PLOG_DBG("Ipc dst is connected.\n");

        //query consumer and queue
        PLOG_DBG("Query queue connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_upConsumer->GetQueueHandle(), microseconds, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "queue");
        PLOG_DBG("Queue is connected.\n");

        PLOG_DBG("Query consumer connection.\n");
        sciErr = NvSciStreamBlockEventQuery(m_upConsumer->GetHandle(), microseconds, &event);
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

    virtual void SetConsumerConfig(const ConsumerConfig& consConfig)
    {
        if (m_upConsumer != nullptr) {
            m_upConsumer->SetConsumerConfig(consConfig);
        }
        m_bUseMailbox = consConfig.bUseMailbox;
    }

    virtual void SetDisplayConfig(const DisplayConfig& dispConfig)
    {
        if (m_upConsumer != nullptr) {
            m_upConsumer->SetDisplayConfig(dispConfig);
        }
    }
  protected:
    virtual const string GetDstChannel() const{
        return m_dstChannel;
    }
    virtual SIPLStatus
    CreateIpcDstAndEndpoint(const string &dstChannel, NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcDst)
    {
        auto status = CFactory::CreateIpcBlock(m_syncModule, m_bufModule, dstChannel.c_str(), false, pEndPoint, pIpcDst);
        PCHK_STATUS_AND_RETURN(status, "CFactory create ipc dst block");
        PLOG_DBG("Dst ipc block is created.\n");

        return NVSIPL_STATUS_OK;
    }

    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler *> &vEventHandlers)
    {
        vEventHandlers.push_back(m_upConsumer.get());
    }
public:
    std::unique_ptr<CConsumer> m_upConsumer = nullptr;

private:
    ConsumerType m_consumerType;
    NvSciStreamBlock m_dstIpcHandle = 0U;
    NvSciIpcEndpoint m_dstIpcEndpoint = 0U;
    string m_dstChannel;
    bool m_bUseMailbox = false;
    std::shared_ptr<COpenWFDController> m_spWFDController{ nullptr };
};

}
}
}

#endif
