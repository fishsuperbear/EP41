/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMFAULTRECEIVESERVICE_PROXY_H
#define MDC_FM_FMFAULTRECEIVESERVICE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmfaultreceiveservice_common.h"
#include <string>

namespace mdc {
namespace fm {
namespace proxy {
namespace events {
    using FaultEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::fm::HzFaultData>;
    using NotifyFaultStateError = ara::com::internal::proxy::event::EventAdapter<::hozon::fm::HzFaultAnalysisEvent>;
    using NotifyFaultEventError = ara::com::internal::proxy::event::EventAdapter<::hozon::fm::HzFaultAnalysisEvent>;
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceFaultEventId = 19654U; //FaultEvent_event_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceNotifyFaultStateErrorId = 63915U; //NotifyFaultStateError_event_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceNotifyFaultEventErrorId = 20316U; //NotifyFaultEventError_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId FmFaultReceiveServiceAlarmReportId = 30975U; //AlarmReport_method_hash
static constexpr ara::com::internal::EntityId FmFaultReceiveServiceAlarmReport_AsyncId = 51475U; //AlarmReport_Async_method_hash
static constexpr ara::com::internal::EntityId FmFaultReceiveServiceGetDataCollectionFileId = 17955U; //GetDataCollectionFile_method_hash
static constexpr ara::com::internal::EntityId FmFaultReceiveServiceRegistIntInterestFaultId = 32336U; //RegistIntInterestFault_method_hash


class AlarmReport {
public:
    using Output = mdc::fm::methods::AlarmReport::Output;

    AlarmReport(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::hozon::fm::HzFaultData& faultMsg)
    {
        return method_(faultMsg);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultData> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultData> method_;
};

class AlarmReport_Async {
public:
    using Output = void;

    AlarmReport_Async(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::hozon::fm::HzFaultData& faultMsg)
    {
        method_(faultMsg);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultData> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultData> method_;
};

class GetDataCollectionFile {
public:
    using Output = mdc::fm::methods::GetDataCollectionFile::Output;

    GetDataCollectionFile(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()()
    {
        return method_();
    }

    ara::com::internal::proxy::method::MethodAdapter<Output> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output> method_;
};

class RegistIntInterestFault {
public:
    using Output = void;

    RegistIntInterestFault(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::hozon::fm::HzFaultItemVector& faultItemVector, const ::hozon::fm::HzFaultClusterVector& faultClusterVector)
    {
        method_(faultItemVector, faultClusterVector);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultItemVector, ::hozon::fm::HzFaultClusterVector> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultItemVector, ::hozon::fm::HzFaultClusterVector> method_;
};
} // namespace methods

class FmFaultReceiveServiceProxy {
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

    virtual ~FmFaultReceiveServiceProxy()
    {
        FaultEvent.UnsetReceiveHandler();
        FaultEvent.Unsubscribe();
        NotifyFaultStateError.UnsetReceiveHandler();
        NotifyFaultStateError.Unsubscribe();
        NotifyFaultEventError.UnsetReceiveHandler();
        NotifyFaultEventError.Unsubscribe();
    }

    explicit FmFaultReceiveServiceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::fm::FmFaultReceiveService::ServiceIdentifier, handle)),
          FaultEvent(proxyAdapter->GetProxy(), events::FmFaultReceiveServiceFaultEventId, proxyAdapter->GetHandle(), ::mdc::fm::FmFaultReceiveService::ServiceIdentifier),
          NotifyFaultStateError(proxyAdapter->GetProxy(), events::FmFaultReceiveServiceNotifyFaultStateErrorId, proxyAdapter->GetHandle(), ::mdc::fm::FmFaultReceiveService::ServiceIdentifier),
          NotifyFaultEventError(proxyAdapter->GetProxy(), events::FmFaultReceiveServiceNotifyFaultEventErrorId, proxyAdapter->GetHandle(), ::mdc::fm::FmFaultReceiveService::ServiceIdentifier),
          AlarmReport(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceAlarmReportId),
          AlarmReport_Async(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceAlarmReport_AsyncId),
          GetDataCollectionFile(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceGetDataCollectionFileId),
          RegistIntInterestFault(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceRegistIntInterestFaultId){
            ara::core::Result<void> resultAlarmReport = proxyAdapter->InitializeMethod<methods::AlarmReport::Output>(methods::FmFaultReceiveServiceAlarmReportId);
            ThrowError(resultAlarmReport);
            ara::core::Result<void> resultAlarmReport_Async = proxyAdapter->InitializeMethod<methods::AlarmReport_Async::Output>(methods::FmFaultReceiveServiceAlarmReport_AsyncId);
            ThrowError(resultAlarmReport_Async);
            ara::core::Result<void> resultGetDataCollectionFile = proxyAdapter->InitializeMethod<methods::GetDataCollectionFile::Output>(methods::FmFaultReceiveServiceGetDataCollectionFileId);
            ThrowError(resultGetDataCollectionFile);
            ara::core::Result<void> resultRegistIntInterestFault = proxyAdapter->InitializeMethod<methods::RegistIntInterestFault::Output>(methods::FmFaultReceiveServiceRegistIntInterestFaultId);
            ThrowError(resultRegistIntInterestFault);
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

    FmFaultReceiveServiceProxy(const FmFaultReceiveServiceProxy&) = delete;
    FmFaultReceiveServiceProxy& operator=(const FmFaultReceiveServiceProxy&) = delete;

    FmFaultReceiveServiceProxy(FmFaultReceiveServiceProxy&&) = default;
    FmFaultReceiveServiceProxy& operator=(FmFaultReceiveServiceProxy&&) = default;
    FmFaultReceiveServiceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          FaultEvent(proxyAdapter->GetProxy(), events::FmFaultReceiveServiceFaultEventId, proxyAdapter->GetHandle(), ::mdc::fm::FmFaultReceiveService::ServiceIdentifier),
          NotifyFaultStateError(proxyAdapter->GetProxy(), events::FmFaultReceiveServiceNotifyFaultStateErrorId, proxyAdapter->GetHandle(), ::mdc::fm::FmFaultReceiveService::ServiceIdentifier),
          NotifyFaultEventError(proxyAdapter->GetProxy(), events::FmFaultReceiveServiceNotifyFaultEventErrorId, proxyAdapter->GetHandle(), ::mdc::fm::FmFaultReceiveService::ServiceIdentifier),
          AlarmReport(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceAlarmReportId),
          AlarmReport_Async(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceAlarmReport_AsyncId),
          GetDataCollectionFile(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceGetDataCollectionFileId),
          RegistIntInterestFault(proxyAdapter->GetProxy(), methods::FmFaultReceiveServiceRegistIntInterestFaultId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::fm::FmFaultReceiveService::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::AlarmReport AlarmReport(preProxyAdapter->GetProxy(), methods::FmFaultReceiveServiceAlarmReportId);
            initResult = preProxyAdapter->InitializeMethod<methods::AlarmReport::Output>(methods::FmFaultReceiveServiceAlarmReportId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::AlarmReport_Async AlarmReport_Async(preProxyAdapter->GetProxy(), methods::FmFaultReceiveServiceAlarmReport_AsyncId);
            initResult = preProxyAdapter->InitializeMethod<methods::AlarmReport_Async::Output>(methods::FmFaultReceiveServiceAlarmReport_AsyncId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetDataCollectionFile GetDataCollectionFile(preProxyAdapter->GetProxy(), methods::FmFaultReceiveServiceGetDataCollectionFileId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetDataCollectionFile::Output>(methods::FmFaultReceiveServiceGetDataCollectionFileId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RegistIntInterestFault RegistIntInterestFault(preProxyAdapter->GetProxy(), methods::FmFaultReceiveServiceRegistIntInterestFaultId);
            initResult = preProxyAdapter->InitializeMethod<methods::RegistIntInterestFault::Output>(methods::FmFaultReceiveServiceRegistIntInterestFaultId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmFaultReceiveService::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmFaultReceiveService::ServiceIdentifier, specifier);
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
    events::FaultEvent FaultEvent;
    events::NotifyFaultStateError NotifyFaultStateError;
    events::NotifyFaultEventError NotifyFaultEventError;
    methods::AlarmReport AlarmReport;
    methods::AlarmReport_Async AlarmReport_Async;
    methods::GetDataCollectionFile GetDataCollectionFile;
    methods::RegistIntInterestFault RegistIntInterestFault;
};
} // namespace proxy
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMFAULTRECEIVESERVICE_PROXY_H
