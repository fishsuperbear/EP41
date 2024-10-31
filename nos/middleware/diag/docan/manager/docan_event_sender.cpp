/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase implement
 */

#include "docan_event_sender.h"
#include "diag/docan/include/docan_listener.h"
#include "diag/docan/taskbase/task_object_def.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanEventSender* DocanEventSender::s_instance = nullptr;
    std::mutex DocanEventSender::s_instance_mutex;

    DocanEventSender::DocanEventSender()
    {
    }

    DocanEventSender::~DocanEventSender()
    {
    }

    DocanEventSender* DocanEventSender::instance()
    {
        std::lock_guard<std::mutex> sync(s_instance_mutex);
        if (nullptr == s_instance) {
            s_instance = new DocanEventSender;
        }
        return s_instance;
    }

    void DocanEventSender::destroy()
    {
        std::lock_guard<std::mutex> sync(s_instance_mutex);
        if (nullptr != s_instance) {
            delete s_instance;
            s_instance = nullptr;
        }
    }

    int32_t DocanEventSender::Init(void)
    {
        return 0;
    }

    int32_t DocanEventSender::Start(void)
    {
        return 0;
    }

    int32_t DocanEventSender::Stop(void)
    {
        return 0;
    }

    int32_t DocanEventSender::Deinit(void)
    {
        return 0;
    }

    int32_t DocanEventSender::sendEvent(uint32_t evtType, uint32_t evtId, int32_t evtArg1, int32_t evtArg2, const std::vector<uint8_t>& evtData)
    {
        return 0;
    }

    int32_t DocanEventSender::sendConfirm(uint16_t reqTa, uint16_t reqSa, uint32_t reqId, uint32_t result)
    {
        return 0;
    }

    int32_t DocanEventSender::sendUdsResponse(const std::string& who, uint16_t reqTa, uint16_t reqSa, uint32_t reqId, uint32_t result, const std::vector<uint8_t>& data)
    {
        DOCAN_LOG_D("sendUdsResponse reqTa: %X, reqSa: %X, reqId: %d, result: %d, uds size: %ld.", reqTa, reqSa, reqId, result, data.size());
        int32_t ret = -1;
        if (m_callbackMap.size() > 0) {
            std::lock_guard<std::mutex> lck(m_sync);
            for (auto it : m_callbackMap) {
                if (it.first == who) {
                    it.second->OnUdsResponse(reqTa, reqSa, reqId, docan_result_t(result), data);
                }
            }
            ret = m_callbackMap.size();
        }
        return ret;
    }

    int32_t DocanEventSender::sendUdsIndication(uint16_t reqTa, uint16_t reqSa, const std::vector<uint8_t>& data)
    {
        DOCAN_LOG_D("sendUdsIndication from reqTa: %X, reqSa: %X, uds size: %ld.", reqTa, reqSa, data.size());
        int32_t ret = -1;
        if (m_callbackMap.size() > 0) {
            std::lock_guard<std::mutex> lck(m_sync);
            for (auto it : m_callbackMap) {
                it.second->OnUdsIndication(reqTa, reqSa, data);
            }
            ret = m_callbackMap.size();
        }
        return ret;
    }

    bool DocanEventSender::isListenerRegistered(const std::string& who)
    {
        std::lock_guard<std::mutex> lck(m_sync);
        for (auto it : m_callbackMap) {
            if (it.first == who) {
                return true;
            }
        }
        return false;
    }

    bool DocanEventSender::isListenerRegistered(const std::string& who, const std::shared_ptr<DocanListener>& listener)
    {
        std::lock_guard<std::mutex> lck(m_sync);
        for (auto it : m_callbackMap) {
            if (it.first == who && it.second == listener) {
                return true;
            }
        }
        return false;
    }

    int32_t DocanEventSender::addListener(const std::string& who, const std::shared_ptr<DocanListener>& listener)
    {
        int32_t ret = -1;
        if (nullptr != listener) {
            std::lock_guard<std::mutex> lck(m_sync);
            m_callbackMap[who] = listener;
            listener->onServiceBind(who);
            ret = m_callbackMap.size();
        }
        return ret;
    }

    int32_t DocanEventSender::removeListener(const std::string& who)
    {
        int32_t ret = -1;
        std::lock_guard<std::mutex> lck(m_sync);
        for (auto it = m_callbackMap.begin(); it != m_callbackMap.end();) {
            if (it->first == who) {
                it->second->onServiceUnbind(who);
                it = m_callbackMap.erase(it);
                ret = m_callbackMap.size();
            }
            else {
                ++it;
            }
        }
        return ret;
    }

    int32_t DocanEventSender::removeListener(const std::shared_ptr<DocanListener>& listener)
    {
        int32_t ret = -1;
        std::lock_guard<std::mutex> lck(m_sync);
        for (auto it = m_callbackMap.begin(); it != m_callbackMap.end();) {
            if (it->second == listener) {
                listener->onServiceBind(it->first);
                it = m_callbackMap.erase(it);
                ret = m_callbackMap.size();
            }
            else {
                ++it;
            }
        }
        return ret;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */