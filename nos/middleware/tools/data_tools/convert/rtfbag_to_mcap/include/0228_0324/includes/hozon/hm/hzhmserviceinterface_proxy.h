/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HM_HZHMSERVICEINTERFACE_PROXY_H
#define HOZON_HM_HZHMSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/hm/hzhmserviceinterface_common.h"
#include <string>

namespace hozon {
namespace hm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceRegistAliveTaskId = 3029U; //RegistAliveTask_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportAliveStatusId = 25182U; //ReportAliveStatus_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceUnRegistAliveTaskId = 6961U; //UnRegistAliveTask_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceRegistDeadlineTaskId = 62516U; //RegistDeadlineTask_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportDeadlineStatusId = 37137U; //ReportDeadlineStatus_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceUnRegistDeadlineTaskId = 37253U; //UnRegistDeadlineTask_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceRegistLogicTaskId = 28627U; //RegistLogicTask_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportLogicCheckpointId = 54299U; //ReportLogicCheckpoint_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceUnRegistLogicTaskId = 63805U; //UnRegistLogicTask_method_hash
static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportProcAliveCheckpointId = 54530U; //ReportProcAliveCheckpoint_method_hash


class RegistAliveTask {
public:
    using Output = hozon::hm::methods::RegistAliveTask::Output;

    RegistAliveTask(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::uint32_t& checkpointId, const ::uint32_t& periodMs, const ::uint32_t& minIndication, const ::uint32_t& maxIndication)
    {
        return method_(processName, checkpointId, periodMs, minIndication, maxIndication);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t, ::uint32_t, ::uint32_t, ::uint32_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t, ::uint32_t, ::uint32_t, ::uint32_t> method_;
};

class ReportAliveStatus {
public:
    using Output = void;

    ReportAliveStatus(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::String& processName, const ::uint32_t& checkpointId, const ::uint8_t& aliveStatus)
    {
        method_(processName, checkpointId, aliveStatus);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t, ::uint8_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t, ::uint8_t> method_;
};

class UnRegistAliveTask {
public:
    using Output = hozon::hm::methods::UnRegistAliveTask::Output;

    UnRegistAliveTask(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::uint32_t& checkpointId)
    {
        return method_(processName, checkpointId);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t> method_;
};

class RegistDeadlineTask {
public:
    using Output = hozon::hm::methods::RegistDeadlineTask::Output;

    RegistDeadlineTask(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::hozon::hm::Transition& transition, const ::uint32_t& deadlineMinMs, const ::uint32_t& deadlineMaxMs)
    {
        return method_(processName, transition, deadlineMinMs, deadlineMaxMs);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transition, ::uint32_t, ::uint32_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transition, ::uint32_t, ::uint32_t> method_;
};

class ReportDeadlineStatus {
public:
    using Output = void;

    ReportDeadlineStatus(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::String& processName, const ::hozon::hm::Transition& transition, const ::uint8_t& deadlineStatus)
    {
        method_(processName, transition, deadlineStatus);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transition, ::uint8_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transition, ::uint8_t> method_;
};

class UnRegistDeadlineTask {
public:
    using Output = hozon::hm::methods::UnRegistDeadlineTask::Output;

    UnRegistDeadlineTask(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::hozon::hm::Transition& transition)
    {
        return method_(processName, transition);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transition> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transition> method_;
};

class RegistLogicTask {
public:
    using Output = hozon::hm::methods::RegistLogicTask::Output;

    RegistLogicTask(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::hozon::hm::Transitions& checkpointIds)
    {
        return method_(processName, checkpointIds);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transitions> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transitions> method_;
};

class ReportLogicCheckpoint {
public:
    using Output = hozon::hm::methods::ReportLogicCheckpoint::Output;

    ReportLogicCheckpoint(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::uint32_t& checkpointId)
    {
        return method_(processName, checkpointId);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t> method_;
};

class UnRegistLogicTask {
public:
    using Output = hozon::hm::methods::UnRegistLogicTask::Output;

    UnRegistLogicTask(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::hozon::hm::Transitions& checkpointIds)
    {
        return method_(processName, checkpointIds);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transitions> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::hozon::hm::Transitions> method_;
};

class ReportProcAliveCheckpoint {
public:
    using Output = hozon::hm::methods::ReportProcAliveCheckpoint::Output;

    ReportProcAliveCheckpoint(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& processName, const ::uint32_t& checkpointId)
    {
        return method_(processName, checkpointId);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint32_t> method_;
};
} // namespace methods

class HzHmServiceInterfaceProxy {
private:
    std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> proxyAdapter;
public:
    using HandleType = vrtf::vcc::api::types::HandleType;
    class ConstructionToken {
    public:
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::proxy::ProxyAdapter>& proxy): ptr(std::move(proxy)){}
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::proxy::ProxyAdapter>&& proxy): ptr(std::move(proxy)){}
        ConstructionToken(ConstructionToken&& other) : ptr(std::move(other.ptr)) {}
        ConstructionToken& operator=(ConstructionToken && other)
        {
            if (&other != this) {
                ptr = std::move(other.ptr);
            }
            return *this;
        }
        ConstructionToken(const ConstructionToken&) = delete;
        ConstructionToken& operator = (const ConstructionToken&) = delete;
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> GetProxyAdapter()
        {
            return std::move(ptr);
        }
    private:
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> ptr;
    };

    virtual ~HzHmServiceInterfaceProxy()
    {
    }

    explicit HzHmServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::hm::HzHmServiceInterface::ServiceIdentifier, handle)),
          RegistAliveTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistAliveTaskId),
          ReportAliveStatus(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportAliveStatusId),
          UnRegistAliveTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistAliveTaskId),
          RegistDeadlineTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistDeadlineTaskId),
          ReportDeadlineStatus(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportDeadlineStatusId),
          UnRegistDeadlineTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId),
          RegistLogicTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistLogicTaskId),
          ReportLogicCheckpoint(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportLogicCheckpointId),
          UnRegistLogicTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistLogicTaskId),
          ReportProcAliveCheckpoint(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId){
            ara::core::Result<void> resultRegistAliveTask = proxyAdapter->InitializeMethod<methods::RegistAliveTask::Output>(methods::HzHmServiceInterfaceRegistAliveTaskId);
            ThrowError(resultRegistAliveTask);
            ara::core::Result<void> resultReportAliveStatus = proxyAdapter->InitializeMethod<methods::ReportAliveStatus::Output>(methods::HzHmServiceInterfaceReportAliveStatusId);
            ThrowError(resultReportAliveStatus);
            ara::core::Result<void> resultUnRegistAliveTask = proxyAdapter->InitializeMethod<methods::UnRegistAliveTask::Output>(methods::HzHmServiceInterfaceUnRegistAliveTaskId);
            ThrowError(resultUnRegistAliveTask);
            ara::core::Result<void> resultRegistDeadlineTask = proxyAdapter->InitializeMethod<methods::RegistDeadlineTask::Output>(methods::HzHmServiceInterfaceRegistDeadlineTaskId);
            ThrowError(resultRegistDeadlineTask);
            ara::core::Result<void> resultReportDeadlineStatus = proxyAdapter->InitializeMethod<methods::ReportDeadlineStatus::Output>(methods::HzHmServiceInterfaceReportDeadlineStatusId);
            ThrowError(resultReportDeadlineStatus);
            ara::core::Result<void> resultUnRegistDeadlineTask = proxyAdapter->InitializeMethod<methods::UnRegistDeadlineTask::Output>(methods::HzHmServiceInterfaceUnRegistDeadlineTaskId);
            ThrowError(resultUnRegistDeadlineTask);
            ara::core::Result<void> resultRegistLogicTask = proxyAdapter->InitializeMethod<methods::RegistLogicTask::Output>(methods::HzHmServiceInterfaceRegistLogicTaskId);
            ThrowError(resultRegistLogicTask);
            ara::core::Result<void> resultReportLogicCheckpoint = proxyAdapter->InitializeMethod<methods::ReportLogicCheckpoint::Output>(methods::HzHmServiceInterfaceReportLogicCheckpointId);
            ThrowError(resultReportLogicCheckpoint);
            ara::core::Result<void> resultUnRegistLogicTask = proxyAdapter->InitializeMethod<methods::UnRegistLogicTask::Output>(methods::HzHmServiceInterfaceUnRegistLogicTaskId);
            ThrowError(resultUnRegistLogicTask);
            ara::core::Result<void> resultReportProcAliveCheckpoint = proxyAdapter->InitializeMethod<methods::ReportProcAliveCheckpoint::Output>(methods::HzHmServiceInterfaceReportProcAliveCheckpointId);
            ThrowError(resultReportProcAliveCheckpoint);
        }

    void ThrowError(const ara::core::Result<void>& result) const
    {
        if (!(result.HasValue())) {
#ifndef NOT_SUPPORT_EXCEPTIONS
            ara::core::ErrorCode errorcode(result.Error());
            throw ara::com::ComException(std::move(errorcode));
#else
            std::cerr << "Error: Not support exception, create proxy failed!"<< std::endl;
#endif
        }
    }

    HzHmServiceInterfaceProxy(const HzHmServiceInterfaceProxy&) = delete;
    HzHmServiceInterfaceProxy& operator=(const HzHmServiceInterfaceProxy&) = delete;

    HzHmServiceInterfaceProxy(HzHmServiceInterfaceProxy&&) = default;
    HzHmServiceInterfaceProxy& operator=(HzHmServiceInterfaceProxy&&) = default;
    HzHmServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          RegistAliveTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistAliveTaskId),
          ReportAliveStatus(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportAliveStatusId),
          UnRegistAliveTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistAliveTaskId),
          RegistDeadlineTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistDeadlineTaskId),
          ReportDeadlineStatus(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportDeadlineStatusId),
          UnRegistDeadlineTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId),
          RegistLogicTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistLogicTaskId),
          ReportLogicCheckpoint(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportLogicCheckpointId),
          UnRegistLogicTask(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistLogicTaskId),
          ReportProcAliveCheckpoint(proxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::hm::HzHmServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::RegistAliveTask RegistAliveTask(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistAliveTaskId);
            initResult = preProxyAdapter->InitializeMethod<methods::RegistAliveTask::Output>(methods::HzHmServiceInterfaceRegistAliveTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReportAliveStatus ReportAliveStatus(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportAliveStatusId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportAliveStatus::Output>(methods::HzHmServiceInterfaceReportAliveStatusId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::UnRegistAliveTask UnRegistAliveTask(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistAliveTaskId);
            initResult = preProxyAdapter->InitializeMethod<methods::UnRegistAliveTask::Output>(methods::HzHmServiceInterfaceUnRegistAliveTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RegistDeadlineTask RegistDeadlineTask(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistDeadlineTaskId);
            initResult = preProxyAdapter->InitializeMethod<methods::RegistDeadlineTask::Output>(methods::HzHmServiceInterfaceRegistDeadlineTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReportDeadlineStatus ReportDeadlineStatus(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportDeadlineStatusId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportDeadlineStatus::Output>(methods::HzHmServiceInterfaceReportDeadlineStatusId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::UnRegistDeadlineTask UnRegistDeadlineTask(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId);
            initResult = preProxyAdapter->InitializeMethod<methods::UnRegistDeadlineTask::Output>(methods::HzHmServiceInterfaceUnRegistDeadlineTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RegistLogicTask RegistLogicTask(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceRegistLogicTaskId);
            initResult = preProxyAdapter->InitializeMethod<methods::RegistLogicTask::Output>(methods::HzHmServiceInterfaceRegistLogicTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReportLogicCheckpoint ReportLogicCheckpoint(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportLogicCheckpointId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportLogicCheckpoint::Output>(methods::HzHmServiceInterfaceReportLogicCheckpointId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::UnRegistLogicTask UnRegistLogicTask(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceUnRegistLogicTaskId);
            initResult = preProxyAdapter->InitializeMethod<methods::UnRegistLogicTask::Output>(methods::HzHmServiceInterfaceUnRegistLogicTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReportProcAliveCheckpoint ReportProcAliveCheckpoint(preProxyAdapter->GetProxy(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportProcAliveCheckpoint::Output>(methods::HzHmServiceInterfaceReportProcAliveCheckpointId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
        } while(false);

        if (result) {
            ConstructionToken token(std::move(preProxyAdapter));
            return ara::core::Result<ConstructionToken>(std::move(token));
        } else {
            ConstructionToken token(std::move(preProxyAdapter));
            ara::core::Result<ConstructionToken> preResult(std::move(token));
            const ara::core::ErrorCode errorcode(initResult.Error());
            preResult.EmplaceError(errorcode);
            return preResult;
        }
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType>& handler,
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::hm::HzHmServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::hm::HzHmServiceInterface::ServiceIdentifier, specifier);
    }

    static void StopFindService(const ara::com::FindServiceHandle& handle)
    {
        ara::com::internal::proxy::ProxyAdapter::StopFindService(handle);
    }

    HandleType GetHandle() const
    {
        return proxyAdapter->GetHandle();
    }
    bool SetEventThreadNumber(const std::uint16_t number, const std::uint16_t queueSize)
    {
        return proxyAdapter->SetEventThreadNumber(number, queueSize);
    }
    methods::RegistAliveTask RegistAliveTask;
    methods::ReportAliveStatus ReportAliveStatus;
    methods::UnRegistAliveTask UnRegistAliveTask;
    methods::RegistDeadlineTask RegistDeadlineTask;
    methods::ReportDeadlineStatus ReportDeadlineStatus;
    methods::UnRegistDeadlineTask UnRegistDeadlineTask;
    methods::RegistLogicTask RegistLogicTask;
    methods::ReportLogicCheckpoint ReportLogicCheckpoint;
    methods::UnRegistLogicTask UnRegistLogicTask;
    methods::ReportProcAliveCheckpoint ReportProcAliveCheckpoint;
};
} // namespace proxy
} // namespace hm
} // namespace hozon

#endif // HOZON_HM_HZHMSERVICEINTERFACE_PROXY_H
