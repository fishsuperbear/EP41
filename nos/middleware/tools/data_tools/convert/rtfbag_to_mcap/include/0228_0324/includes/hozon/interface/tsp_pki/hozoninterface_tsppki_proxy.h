/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_PROXY_H
#define HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/tsp_pki/hozoninterface_tsppki_common.h"
#include <string>

namespace hozon {
namespace interface {
namespace tsp_pki {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HozonInterface_TspPkiRequestHdUuidId = 60529U; //RequestHdUuid_method_hash
static constexpr ara::com::internal::EntityId HozonInterface_TspPkiRequestUploadTokenId = 39210U; //RequestUploadToken_method_hash
static constexpr ara::com::internal::EntityId HozonInterface_TspPkiRequestRemoteConfigId = 27288U; //RequestRemoteConfig_method_hash
static constexpr ara::com::internal::EntityId HozonInterface_TspPkiReadPkiStatusId = 30124U; //ReadPkiStatus_method_hash


class RequestHdUuid {
public:
    using Output = hozon::interface::tsp_pki::methods::RequestHdUuid::Output;

    RequestHdUuid(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class RequestUploadToken {
public:
    using Output = hozon::interface::tsp_pki::methods::RequestUploadToken::Output;

    RequestUploadToken(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class RequestRemoteConfig {
public:
    using Output = hozon::interface::tsp_pki::methods::RequestRemoteConfig::Output;

    RequestRemoteConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class ReadPkiStatus {
public:
    using Output = hozon::interface::tsp_pki::methods::ReadPkiStatus::Output;

    ReadPkiStatus(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class HozonInterface_TspPkiProxy {
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

    virtual ~HozonInterface_TspPkiProxy()
    {
    }

    explicit HozonInterface_TspPkiProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, handle)),
          RequestHdUuid(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestHdUuidId),
          RequestUploadToken(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestUploadTokenId),
          RequestRemoteConfig(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestRemoteConfigId),
          ReadPkiStatus(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiReadPkiStatusId){
            ara::core::Result<void> resultRequestHdUuid = proxyAdapter->InitializeMethod<methods::RequestHdUuid::Output>(methods::HozonInterface_TspPkiRequestHdUuidId);
            ThrowError(resultRequestHdUuid);
            ara::core::Result<void> resultRequestUploadToken = proxyAdapter->InitializeMethod<methods::RequestUploadToken::Output>(methods::HozonInterface_TspPkiRequestUploadTokenId);
            ThrowError(resultRequestUploadToken);
            ara::core::Result<void> resultRequestRemoteConfig = proxyAdapter->InitializeMethod<methods::RequestRemoteConfig::Output>(methods::HozonInterface_TspPkiRequestRemoteConfigId);
            ThrowError(resultRequestRemoteConfig);
            ara::core::Result<void> resultReadPkiStatus = proxyAdapter->InitializeMethod<methods::ReadPkiStatus::Output>(methods::HozonInterface_TspPkiReadPkiStatusId);
            ThrowError(resultReadPkiStatus);
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

    HozonInterface_TspPkiProxy(const HozonInterface_TspPkiProxy&) = delete;
    HozonInterface_TspPkiProxy& operator=(const HozonInterface_TspPkiProxy&) = delete;

    HozonInterface_TspPkiProxy(HozonInterface_TspPkiProxy&&) = default;
    HozonInterface_TspPkiProxy& operator=(HozonInterface_TspPkiProxy&&) = default;
    HozonInterface_TspPkiProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          RequestHdUuid(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestHdUuidId),
          RequestUploadToken(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestUploadTokenId),
          RequestRemoteConfig(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestRemoteConfigId),
          ReadPkiStatus(proxyAdapter->GetProxy(), methods::HozonInterface_TspPkiReadPkiStatusId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::RequestHdUuid RequestHdUuid(preProxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestHdUuidId);
            initResult = preProxyAdapter->InitializeMethod<methods::RequestHdUuid::Output>(methods::HozonInterface_TspPkiRequestHdUuidId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RequestUploadToken RequestUploadToken(preProxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestUploadTokenId);
            initResult = preProxyAdapter->InitializeMethod<methods::RequestUploadToken::Output>(methods::HozonInterface_TspPkiRequestUploadTokenId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RequestRemoteConfig RequestRemoteConfig(preProxyAdapter->GetProxy(), methods::HozonInterface_TspPkiRequestRemoteConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::RequestRemoteConfig::Output>(methods::HozonInterface_TspPkiRequestRemoteConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReadPkiStatus ReadPkiStatus(preProxyAdapter->GetProxy(), methods::HozonInterface_TspPkiReadPkiStatusId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReadPkiStatus::Output>(methods::HozonInterface_TspPkiReadPkiStatusId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, specifier);
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
    methods::RequestHdUuid RequestHdUuid;
    methods::RequestUploadToken RequestUploadToken;
    methods::RequestRemoteConfig RequestRemoteConfig;
    methods::ReadPkiStatus ReadPkiStatus;
};
} // namespace proxy
} // namespace tsp_pki
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_PROXY_H