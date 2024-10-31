// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CCHANNEL_H
#define CCHANNEL_H

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <thread>
#include <atomic>

#include "sensor/nvs_consumer/CCmdLineParser.hpp"
#include "sensor/nvs_consumer/CUtils.hpp"
#include "sensor/nvs_consumer/CClientCommon.hpp"

#include "nvscibuf.h"
#include "NvSIPLCamera.hpp"

using namespace std;
using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace desay { 

class CChannel
{
public:
    CChannel() = delete;
    CChannel(string name, NvSciBufModule& bufMod, NvSciSyncModule& syncMod, SensorInfo *pSensorInfo)
    {
        m_name = name + std::to_string(pSensorInfo->id);
        m_bufModule = bufMod;
        m_syncModule = syncMod;
        m_pSensorInfo = pSensorInfo;
    }
    virtual ~CChannel() {}

    virtual SIPLStatus CreateBlocks(CProfiler *pProfiler) = 0;
    virtual SIPLStatus Connect(void) = 0;
    virtual SIPLStatus InitBlocks(void) = 0;
    virtual SIPLStatus Deinit(void) = 0;
    virtual SIPLStatus Init(void)
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Reconcile(void)
    {
        PLOG_DBG("Reconcile.\n");

        m_bRunning = true;
        std::vector<std::unique_ptr<std::thread>> vupThreads;
        std::vector<CEventHandler*> vEventThreadHandlers;

        GetEventThreadHandlers(false, vEventThreadHandlers);
        for (const auto& pEventHandler : vEventThreadHandlers) {
            vupThreads.push_back(std::make_unique<std::thread>(EventThreadFunc, pEventHandler, this));
        }

        for (auto& upThread: vupThreads) {
            if (upThread != nullptr) {
                upThread->join();
                upThread.reset();
                LOG_DBG("upThread->join.\n");
            }
        }
        if (!m_bRunning) {
            PLOG_ERR("SetupStream failed.\n");
            return NVSIPL_STATUS_ERROR;
        }

        PLOG_DBG("SetupStream succeed.\n");
        return NVSIPL_STATUS_OK;
    }

    void Start(void)
    {
        std::vector<CEventHandler*> vEventThreadHandlers;

        PLOG_DBG("Start.\n");

        m_bRunning = true;
        GetEventThreadHandlers(true, vEventThreadHandlers);
        for (const auto& pEventHandler : vEventThreadHandlers) {
            m_vupThreads.push_back(std::make_unique<std::thread>(EventThreadFunc, pEventHandler, this));
        }
    }

    void Stop(void)
    {
        PLOG_DBG("Stop.\n");

        m_bRunning = false;
        for (auto& upThread: m_vupThreads) {
            if (upThread != nullptr) {
                upThread->join();
                upThread.reset();
            }
        }
        PLOG_DBG("Stop, all threads exit now.\n");
    }

    /* The per-thread loop function for each block */
    static void EventThreadFunc(CEventHandler* pEventHandler, CChannel* pChannel)
    {
        uint32_t timeouts = 0U;
        EventStatus eventStatus = EVENT_STATUS_OK;

        string threadName = pEventHandler->GetName();
        pthread_setname_np(pthread_self(), threadName.c_str());

        /* Simple loop, waiting for events on the block until the block is done */
        while (pChannel->m_bRunning) {
            eventStatus = pEventHandler->HandleEvents();
            if (eventStatus == EVENT_STATUS_TIMED_OUT) {
                // if query timeouts - keep waiting for event until wait threshold is reached
                if (timeouts < MAX_QUERY_TIMEOUTS) {
                    timeouts++;
                    continue;
                }
                LOG_WARN((pEventHandler->GetName() + ": HandleEvents() seems to be taking forever!\n").c_str());
            } else if (eventStatus == EVENT_STATUS_OK) {
                timeouts = 0U;
                continue;
            } else if (eventStatus == EVENT_STATUS_COMPLETE) {
                break;
            } else {
               pChannel->m_bRunning = false;
               break;
            }
        }
    }
    virtual void SetConsumerConfig(const ConsumerConfig& consConfig) {}
    virtual void SetDisplayConfig(const DisplayConfig& dispConfig) {}

protected:
    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler*>& vEventHandlers) = 0;

    string m_name;
    NvSciBufModule m_bufModule = nullptr;
    NvSciSyncModule m_syncModule = nullptr;
    SensorInfo *m_pSensorInfo = nullptr;

private:
    atomic<bool> m_bRunning;
    vector<unique_ptr<thread>> m_vupThreads;
};

}
}
}
#endif
