/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMMCUREPORTFAULTSERVICE_PROXY_H
#define MDC_FM_FMMCUREPORTFAULTSERVICE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmmcureportfaultservice_common.h"
#include <string>

namespace mdc {
namespace fm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId FmMcuReportFaultServiceMcuReportFaultId = 63850U; //McuReportFault_method_hash


class McuReportFault {
public:
    using Output = mdc::fm::methods::McuReportFault::Output;

    McuReportFault(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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
} // namespace methods

class FmMcuReportFaultServiceProxy {
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

    virtual ~FmMcuReportFaultServiceProxy()
    {
    }

    explicit FmMcuReportFaultServiceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::fm::FmMcuReportFaultService::ServiceIdentifier, handle)),
          McuReportFault(proxyAdapter->GetProxy(), methods::FmMcuReportFaultServiceMcuReportFaultId){
            ara::core::Result<void> resultMcuReportFault = proxyAdapter->InitializeMethod<methods::McuReportFault::Output>(methods::FmMcuReportFaultServiceMcuReportFaultId);
            ThrowError(resultMcuReportFault);
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

    FmMcuReportFaultServiceProxy(const FmMcuReportFaultServiceProxy&) = delete;
    FmMcuReportFaultServiceProxy& operator=(const FmMcuReportFaultServiceProxy&) = delete;

    FmMcuReportFaultServiceProxy(FmMcuReportFaultServiceProxy&&) = default;
    FmMcuReportFaultServiceProxy& operator=(FmMcuReportFaultServiceProxy&&) = default;
    FmMcuReportFaultServiceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          McuReportFault(proxyAdapter->GetProxy(), methods::FmMcuReportFaultServiceMcuReportFaultId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::fm::FmMcuReportFaultService::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::McuReportFault McuReportFault(preProxyAdapter->GetProxy(), methods::FmMcuReportFaultServiceMcuReportFaultId);
            initResult = preProxyAdapter->InitializeMethod<methods::McuReportFault::Output>(methods::FmMcuReportFaultServiceMcuReportFaultId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmMcuReportFaultService::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmMcuReportFaultService::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmMcuReportFaultService::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmMcuReportFaultService::ServiceIdentifier, specifier);
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
    methods::McuReportFault McuReportFault;
};
} // namespace proxy
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMMCUREPORTFAULTSERVICE_PROXY_H
