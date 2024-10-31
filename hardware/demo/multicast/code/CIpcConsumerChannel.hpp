// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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

using namespace std;
using namespace nvsipl;

class CIpcConsumerChannel: public CChannel
{
public:
    CIpcConsumerChannel() = delete;
    CIpcConsumerChannel(NvSciBufModule& bufMod,
        NvSciSyncModule& syncMod, SensorInfo *pSensorInfo, ConsumerType consumerType) :
        CChannel("IpcConsChan", bufMod, syncMod, pSensorInfo)
    {
        m_consumerType = consumerType;
        m_dstChannel = "nvscistream_" + std::to_string(pSensorInfo->id * MAX_NUM_CONSUMERS * 2 + 2 * consumerType + 1);
        m_dstIpcHandle = 0U;
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
    }

    SIPLStatus CreateBlocks(CProfiler *pProfiler)
    {
        PLOG_DBG("CreateBlocks.\n");

        m_upConsumer = CFactory::CreateConsumer(m_consumerType, m_pSensorInfo, m_bUseMailbox);
        PCHK_PTR_AND_RETURN(m_upConsumer, "CFactory::CreateConsumer");
        m_upConsumer->SetProfiler(pProfiler);
        PLOG_DBG((m_upConsumer->GetName() + " is created.\n").c_str());

        auto status = CFactory::CreateIpcBlock(m_syncModule, m_bufModule, m_dstChannel.c_str(), false, &m_dstIpcHandle);
        PCHK_STATUS_AND_RETURN(status, "CFactory::CreateIpcBlock");
        PLOG_DBG("Dst ipc block is created.\n");

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

    virtual void SetConsumerConfig(const ConsumerConfig& consConfig)
    {
        if (m_upConsumer != nullptr) {
            m_upConsumer->SetConsumerConfig(consConfig);
        }
        m_bUseMailbox = consConfig.bUseMailbox;
    }

protected:
    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler*>& vEventHandlers)
    {
        vEventHandlers.push_back(m_upConsumer.get());
    }

private:
    ConsumerType m_consumerType;
    std::unique_ptr<CConsumer> m_upConsumer = nullptr;
    NvSciStreamBlock m_dstIpcHandle;
    string m_dstChannel;
    bool m_bUseMailbox = false;
};

#endif
