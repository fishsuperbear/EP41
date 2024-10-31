/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMREPORTEVENTSERVICE_PROXY_H
#define MDC_FM_FMREPORTEVENTSERVICE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmreporteventservice_common.h"
#include <string>

namespace mdc {
namespace fm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId FmReportEventServiceReportFaultId = 14388U; //ReportFault_method_hash
static constexpr ara::com::internal::EntityId FmReportEventServiceReportCheckPointId = 3322U; //ReportCheckPoint_method_hash
static constexpr ara::com::internal::EntityId FmReportEventServiceReportProcStateId = 24315U; //ReportProcState_method_hash


class ReportFault {
public:
    using Output = mdc::fm::methods::ReportFault::Output;

    ReportFault(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::mdc::fm::FmFaultData& faultData)
    {
        return method_(faultData);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::fm::FmFaultData> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::fm::FmFaultData> method_;
};

class ReportCheckPoint {
public:
    using Output = mdc::fm::methods::ReportCheckPoint::Output;

    ReportCheckPoint(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& procName)
    {
        return method_(procName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> method_;
};

class ReportProcState {
public:
    using Output = mdc::fm::methods::ReportProcState::Output;

    ReportProcState(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& procName, const ::uint8_t& state)
    {
        return method_(procName, state);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint8_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::uint8_t> method_;
};
} // namespace methods

class FmReportEventServiceProxy {
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

    virtual ~FmReportEventServiceProxy()
    {
    }

    explicit FmReportEventServiceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::fm::FmReportEventService::ServiceIdentifier, handle)),
          ReportFault(proxyAdapter->GetProxy(), methods::FmReportEventServiceReportFaultId),
          ReportCheckPoint(proxyAdapter->GetProxy(), methods::FmReportEventServiceReportCheckPointId),
          ReportProcState(proxyAdapter->GetProxy(), methods::FmReportEventServiceReportProcStateId){
            ara::core::Result<void> resultReportFault = proxyAdapter->InitializeMethod<methods::ReportFault::Output>(methods::FmReportEventServiceReportFaultId);
            ThrowError(resultReportFault);
            ara::core::Result<void> resultReportCheckPoint = proxyAdapter->InitializeMethod<methods::ReportCheckPoint::Output>(methods::FmReportEventServiceReportCheckPointId);
            ThrowError(resultReportCheckPoint);
            ara::core::Result<void> resultReportProcState = proxyAdapter->InitializeMethod<methods::ReportProcState::Output>(methods::FmReportEventServiceReportProcStateId);
            ThrowError(resultReportProcState);
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

    FmReportEventServiceProxy(const FmReportEventServiceProxy&) = delete;
    FmReportEventServiceProxy& operator=(const FmReportEventServiceProxy&) = delete;

    FmReportEventServiceProxy(FmReportEventServiceProxy&&) = default;
    FmReportEventServiceProxy& operator=(FmReportEventServiceProxy&&) = default;
    FmReportEventServiceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          ReportFault(proxyAdapter->GetProxy(), methods::FmReportEventServiceReportFaultId),
          ReportCheckPoint(proxyAdapter->GetProxy(), methods::FmReportEventServiceReportCheckPointId),
          ReportProcState(proxyAdapter->GetProxy(), methods::FmReportEventServiceReportProcStateId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::fm::FmReportEventService::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::ReportFault ReportFault(preProxyAdapter->GetProxy(), methods::FmReportEventServiceReportFaultId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportFault::Output>(methods::FmReportEventServiceReportFaultId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReportCheckPoint ReportCheckPoint(preProxyAdapter->GetProxy(), methods::FmReportEventServiceReportCheckPointId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportCheckPoint::Output>(methods::FmReportEventServiceReportCheckPointId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReportProcState ReportProcState(preProxyAdapter->GetProxy(), methods::FmReportEventServiceReportProcStateId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReportProcState::Output>(methods::FmReportEventServiceReportProcStateId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmReportEventService::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmReportEventService::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmReportEventService::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmReportEventService::ServiceIdentifier, specifier);
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
    methods::ReportFault ReportFault;
    methods::ReportCheckPoint ReportCheckPoint;
    methods::ReportProcState ReportProcState;
};
} // namespace proxy
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMREPORTEVENTSERVICE_PROXY_H
