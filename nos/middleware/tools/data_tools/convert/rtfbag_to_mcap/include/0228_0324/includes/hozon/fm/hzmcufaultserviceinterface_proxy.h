/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_HZMCUFAULTSERVICEINTERFACE_PROXY_H
#define HOZON_FM_HZMCUFAULTSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/fm/hzmcufaultserviceinterface_common.h"
#include <string>

namespace hozon {
namespace fm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HzMCUFaultServiceInterfaceFaultReportId = 24102U; //FaultReport_method_hash
static constexpr ara::com::internal::EntityId HzMCUFaultServiceInterfaceFaultToHMIId = 18044U; //FaultToHMI_method_hash


class FaultReport {
public:
    using Output = hozon::fm::methods::FaultReport::Output;

    FaultReport(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::hozon::fm::HzFaultEventToMCU& FaultData)
    {
        return method_(FaultData);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultEventToMCU> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::fm::HzFaultEventToMCU> method_;
};

class FaultToHMI {
public:
    using Output = void;

    FaultToHMI(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::uint64_t& FaultToHMIData)
    {
        method_(FaultToHMIData);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint64_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint64_t> method_;
};
} // namespace methods

class HzMCUFaultServiceInterfaceProxy {
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

    virtual ~HzMCUFaultServiceInterfaceProxy()
    {
    }

    explicit HzMCUFaultServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::fm::HzMCUFaultServiceInterface::ServiceIdentifier, handle)),
          FaultReport(proxyAdapter->GetProxy(), methods::HzMCUFaultServiceInterfaceFaultReportId),
          FaultToHMI(proxyAdapter->GetProxy(), methods::HzMCUFaultServiceInterfaceFaultToHMIId){
            ara::core::Result<void> resultFaultReport = proxyAdapter->InitializeMethod<methods::FaultReport::Output>(methods::HzMCUFaultServiceInterfaceFaultReportId);
            ThrowError(resultFaultReport);
            ara::core::Result<void> resultFaultToHMI = proxyAdapter->InitializeMethod<methods::FaultToHMI::Output>(methods::HzMCUFaultServiceInterfaceFaultToHMIId);
            ThrowError(resultFaultToHMI);
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

    HzMCUFaultServiceInterfaceProxy(const HzMCUFaultServiceInterfaceProxy&) = delete;
    HzMCUFaultServiceInterfaceProxy& operator=(const HzMCUFaultServiceInterfaceProxy&) = delete;

    HzMCUFaultServiceInterfaceProxy(HzMCUFaultServiceInterfaceProxy&&) = default;
    HzMCUFaultServiceInterfaceProxy& operator=(HzMCUFaultServiceInterfaceProxy&&) = default;
    HzMCUFaultServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          FaultReport(proxyAdapter->GetProxy(), methods::HzMCUFaultServiceInterfaceFaultReportId),
          FaultToHMI(proxyAdapter->GetProxy(), methods::HzMCUFaultServiceInterfaceFaultToHMIId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::fm::HzMCUFaultServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::FaultReport FaultReport(preProxyAdapter->GetProxy(), methods::HzMCUFaultServiceInterfaceFaultReportId);
            initResult = preProxyAdapter->InitializeMethod<methods::FaultReport::Output>(methods::HzMCUFaultServiceInterfaceFaultReportId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::FaultToHMI FaultToHMI(preProxyAdapter->GetProxy(), methods::HzMCUFaultServiceInterfaceFaultToHMIId);
            initResult = preProxyAdapter->InitializeMethod<methods::FaultToHMI::Output>(methods::HzMCUFaultServiceInterfaceFaultToHMIId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::fm::HzMCUFaultServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::fm::HzMCUFaultServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::fm::HzMCUFaultServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::fm::HzMCUFaultServiceInterface::ServiceIdentifier, specifier);
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
    methods::FaultReport FaultReport;
    methods::FaultToHMI FaultToHMI;
};
} // namespace proxy
} // namespace fm
} // namespace hozon

#endif // HOZON_FM_HZMCUFAULTSERVICEINTERFACE_PROXY_H
