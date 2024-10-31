/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TSYNC_TSYNCAPSERVICEINTERFACE_PROXY_H
#define ARA_TSYNC_TSYNCAPSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/tsync/tsyncapserviceinterface_common.h"
#include <string>

namespace ara {
namespace tsync {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId TsyncApServiceInterfaceGetDataPlaneStatusFlagId = 18085U; //GetDataPlaneStatusFlag_method_hash
static constexpr ara::com::internal::EntityId TsyncApServiceInterfaceGetManagePlaneStatusFlagId = 47341U; //GetManagePlaneStatusFlag_method_hash


class GetDataPlaneStatusFlag {
public:
    using Output = ara::tsync::methods::GetDataPlaneStatusFlag::Output;

    GetDataPlaneStatusFlag(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetManagePlaneStatusFlag {
public:
    using Output = ara::tsync::methods::GetManagePlaneStatusFlag::Output;

    GetManagePlaneStatusFlag(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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
} // namespace methods

class TsyncApServiceInterfaceProxy {
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

    virtual ~TsyncApServiceInterfaceProxy()
    {
    }

    explicit TsyncApServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::ara::tsync::TsyncApServiceInterface::ServiceIdentifier, handle)),
          GetDataPlaneStatusFlag(proxyAdapter->GetProxy(), methods::TsyncApServiceInterfaceGetDataPlaneStatusFlagId),
          GetManagePlaneStatusFlag(proxyAdapter->GetProxy(), methods::TsyncApServiceInterfaceGetManagePlaneStatusFlagId){
            ara::core::Result<void> resultGetDataPlaneStatusFlag = proxyAdapter->InitializeMethod<methods::GetDataPlaneStatusFlag::Output>(methods::TsyncApServiceInterfaceGetDataPlaneStatusFlagId);
            ThrowError(resultGetDataPlaneStatusFlag);
            ara::core::Result<void> resultGetManagePlaneStatusFlag = proxyAdapter->InitializeMethod<methods::GetManagePlaneStatusFlag::Output>(methods::TsyncApServiceInterfaceGetManagePlaneStatusFlagId);
            ThrowError(resultGetManagePlaneStatusFlag);
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

    TsyncApServiceInterfaceProxy(const TsyncApServiceInterfaceProxy&) = delete;
    TsyncApServiceInterfaceProxy& operator=(const TsyncApServiceInterfaceProxy&) = delete;

    TsyncApServiceInterfaceProxy(TsyncApServiceInterfaceProxy&&) = default;
    TsyncApServiceInterfaceProxy& operator=(TsyncApServiceInterfaceProxy&&) = default;
    TsyncApServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          GetDataPlaneStatusFlag(proxyAdapter->GetProxy(), methods::TsyncApServiceInterfaceGetDataPlaneStatusFlagId),
          GetManagePlaneStatusFlag(proxyAdapter->GetProxy(), methods::TsyncApServiceInterfaceGetManagePlaneStatusFlagId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::ara::tsync::TsyncApServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::GetDataPlaneStatusFlag GetDataPlaneStatusFlag(preProxyAdapter->GetProxy(), methods::TsyncApServiceInterfaceGetDataPlaneStatusFlagId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetDataPlaneStatusFlag::Output>(methods::TsyncApServiceInterfaceGetDataPlaneStatusFlagId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetManagePlaneStatusFlag GetManagePlaneStatusFlag(preProxyAdapter->GetProxy(), methods::TsyncApServiceInterfaceGetManagePlaneStatusFlagId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetManagePlaneStatusFlag::Output>(methods::TsyncApServiceInterfaceGetManagePlaneStatusFlagId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::tsync::TsyncApServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::tsync::TsyncApServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::tsync::TsyncApServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::tsync::TsyncApServiceInterface::ServiceIdentifier, specifier);
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
    methods::GetDataPlaneStatusFlag GetDataPlaneStatusFlag;
    methods::GetManagePlaneStatusFlag GetManagePlaneStatusFlag;
};
} // namespace proxy
} // namespace tsync
} // namespace ara

#endif // ARA_TSYNC_TSYNCAPSERVICEINTERFACE_PROXY_H
