/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanService implement
 */

#include "diag/docan/include/docan_service.h"
#include "docan_service_impl.h"
#include "diag/docan/include/docan_listener.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {


    DocanService* DocanService::s_instance = nullptr;
    /**
     * @brief Constructor.
     */
    DocanService::DocanService()
        : m_impl(nullptr)
    {
        m_impl = new DocanServiceImpl();
        s_instance = this;
        if (nullptr != m_impl) {
            // m_impl->registerService();
        }
    }

    DocanService::~DocanService()
    {
        DOCAN_LOG_I("~DocanService Destructor");
        s_instance = nullptr;
        if (nullptr != m_impl) {
            delete m_impl;
            m_impl = nullptr;
        }
    }

    DocanService* DocanService::GetDocanService()
    {
        return s_instance;
    }

    int32_t DocanService::Init()
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->Init();
    }

    int32_t DocanService::Start()
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->Start();
    }

    int32_t DocanService::Stop()
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->Stop();
    }

    int32_t DocanService::Deinit()
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->Deinit();
    }

    int32_t DocanService::registerListener(const std::string& who, const std::shared_ptr<DocanListener>& listener)
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->registerListener(who, listener);
    }

    int32_t DocanService::unregisterListener(const std::string& who)
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->unregisterListener(who);
    }

    int32_t DocanService::UdsRequest(const std::string& who, uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds)
    {
        int32_t ret = -1;
        if (nullptr == m_impl) {
            return ret;
        }
        return m_impl->UdsRequest(who, reqSa, reqTa, uds);
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */