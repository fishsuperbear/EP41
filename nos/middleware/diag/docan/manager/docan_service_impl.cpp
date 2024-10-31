/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase implement
 */

#include "docan_service_impl.h"
#include "diag/docan/include/docan_listener.h"
#include "diag/docan/config/docan_config.h"
#include "diag/docan/log/docan_log.h"

#include "docan_state_manager.h"
#include "docan_sys_interface.h"
#include "docan_event_sender.h"
#include "docan_event_receiver.h"
#include "docan_task_runner.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanServiceImpl::DocanServiceImpl()
    {
        DocanLogger::GetInstance().CreateLogger("docan");
        DOCAN_LOG_D("Docan serviceImpl construct()!");
        DocanConfig::instance();
        DocanStateManager::instance();
        DocanSysInterface::instance();
        DocanEventSender::instance();
        DocanEventReceiver::instance();
        DocanTaskRunner::instance();
    }

    DocanServiceImpl::~DocanServiceImpl()
    {
        DOCAN_LOG_D("Docan serviceImpl destructor()~~!");
        DocanTaskRunner::destroy();
        DocanEventReceiver::destroy();
        DocanEventSender::destroy();
        DocanSysInterface::destroy();
        DocanStateManager::destroy();
        DocanConfig::destroy();
    }

    int32_t DocanServiceImpl::Init()
    {
        DOCAN_LOG_D("Docan serviceImpl init()");
        DocanConfig::instance()->Init();
        DocanStateManager::instance()->Init();
        DocanSysInterface::instance()->Init();
        DocanEventSender::instance()->Init();
        DocanEventReceiver::instance()->Init();
        DocanTaskRunner::instance()->Init();
        return 0;
    }


    int32_t DocanServiceImpl::Start()
    {
        DOCAN_LOG_D("Docan serviceImpl start()");
        DocanConfig::instance()->Start();
        DocanStateManager::instance()->Start();
        DocanSysInterface::instance()->Start();
        DocanEventSender::instance()->Start();
        DocanEventReceiver::instance()->Start();
        DocanTaskRunner::instance()->Start();
        return 0;
    }

    int32_t DocanServiceImpl::Stop()
    {
        DOCAN_LOG_D("Docan serviceImpl stop()");
        DocanConfig::instance()->Stop();
        DocanStateManager::instance()->Stop();
        DocanSysInterface::instance()->Stop();
        DocanEventSender::instance()->Stop();
        DocanEventReceiver::instance()->Stop();
        DocanTaskRunner::instance()->Stop();
        return 0;
    }

    int32_t DocanServiceImpl::Deinit()
    {
        DOCAN_LOG_D("Docan serviceImpl deinit()");
        DocanConfig::instance()->Deinit();
        DocanStateManager::instance()->Deinit();
        DocanSysInterface::instance()->Deinit();
        DocanEventSender::instance()->Deinit();
        DocanEventReceiver::instance()->Deinit();
        DocanTaskRunner::instance()->Deinit();
        return 0;
    }

    int32_t DocanServiceImpl::registerListener(const std::string& who, const std::shared_ptr<DocanListener>& listener)
    {
        int32_t ret = -1;
        if ("" != who && nullptr != listener) {
            ret =DocanEventSender::instance()->addListener(who, listener);
        }
        return ret;
    }

    int32_t DocanServiceImpl::unregisterListener(const std::string& who)
    {
        int32_t ret = -1;
        if ("" != who) {
            ret = DocanEventSender::instance()->removeListener(who);
        }
        return ret;
    }

    int32_t DocanServiceImpl::UdsRequest(const std::string& who, uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds)
    {
        int32_t reqId = -1;
        if (DocanEventSender::instance()->isListenerRegistered(who)
            || DIAG_FUNCTIONAL_ADDR_DOCAN == reqTa) {
            reqId = DocanTaskRunner::instance()->UdsRequest(who, reqSa, reqTa, uds);
        }
        else {
            // listener is not registered while not functional request.
            reqId = 0;
        }
        DOCAN_LOG_D("UdsRequest who: %s, reqSa: %X, reqTa: %X, uds size: %ld, reqId: %d.", who.c_str(), reqSa, reqTa, uds.size(), reqId);
        return reqId;
    }


} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
