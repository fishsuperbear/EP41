/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_VO_VOCONFIGSERVICEINTERFACE_PROXY_H
#define MDC_VO_VOCONFIGSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/vo/voconfigserviceinterface_common.h"
#include <string>

namespace mdc {
namespace vo {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId VoConfigServiceInterfaceGetVoConfigId = 56223U; //GetVoConfig_method_hash
static constexpr ara::com::internal::EntityId VoConfigServiceInterfaceGetVoChnAttrId = 9804U; //GetVoChnAttr_method_hash
static constexpr ara::com::internal::EntityId VoConfigServiceInterfaceSetVoChnAttrId = 44618U; //SetVoChnAttr_method_hash
static constexpr ara::com::internal::EntityId VoConfigServiceInterfaceSetVoSourceId = 22248U; //SetVoSource_method_hash
static constexpr ara::com::internal::EntityId VoConfigServiceInterfaceGetVoSourceId = 30687U; //GetVoSource_method_hash
static constexpr ara::com::internal::EntityId VoConfigServiceInterfaceSetChnDisplayAttrId = 17796U; //SetChnDisplayAttr_method_hash


class GetVoConfig {
public:
    using Output = mdc::vo::methods::GetVoConfig::Output;

    GetVoConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetVoChnAttr {
public:
    using Output = mdc::vo::methods::GetVoChnAttr::Output;

    GetVoChnAttr(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& chnId)
    {
        return method_(chnId);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t> method_;
};

class SetVoChnAttr {
public:
    using Output = mdc::vo::methods::SetVoChnAttr::Output;

    SetVoChnAttr(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& chnId, const ::mdc::vo::ChnAttr& chnAttr)
    {
        return method_(chnId, chnAttr);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::mdc::vo::ChnAttr> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::mdc::vo::ChnAttr> method_;
};

class SetVoSource {
public:
    using Output = mdc::vo::methods::SetVoSource::Output;

    SetVoSource(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& chnId, const ::mdc::vo::VoEnum& videoSource)
    {
        return method_(chnId, videoSource);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::mdc::vo::VoEnum> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::mdc::vo::VoEnum> method_;
};

class GetVoSource {
public:
    using Output = mdc::vo::methods::GetVoSource::Output;

    GetVoSource(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& chnId)
    {
        return method_(chnId);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t> method_;
};

class SetChnDisplayAttr {
public:
    using Output = mdc::vo::methods::SetChnDisplayAttr::Output;

    SetChnDisplayAttr(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& chnId, const ::mdc::vo::DisplayAttrEnum& setting)
    {
        return method_(chnId, setting);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::mdc::vo::DisplayAttrEnum> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::mdc::vo::DisplayAttrEnum> method_;
};
} // namespace methods

class VoConfigServiceInterfaceProxy {
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

    virtual ~VoConfigServiceInterfaceProxy()
    {
    }

    explicit VoConfigServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::vo::VoConfigServiceInterface::ServiceIdentifier, handle)),
          GetVoConfig(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoConfigId),
          GetVoChnAttr(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoChnAttrId),
          SetVoChnAttr(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetVoChnAttrId),
          SetVoSource(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetVoSourceId),
          GetVoSource(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoSourceId),
          SetChnDisplayAttr(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetChnDisplayAttrId){
            ara::core::Result<void> resultGetVoConfig = proxyAdapter->InitializeMethod<methods::GetVoConfig::Output>(methods::VoConfigServiceInterfaceGetVoConfigId);
            ThrowError(resultGetVoConfig);
            ara::core::Result<void> resultGetVoChnAttr = proxyAdapter->InitializeMethod<methods::GetVoChnAttr::Output>(methods::VoConfigServiceInterfaceGetVoChnAttrId);
            ThrowError(resultGetVoChnAttr);
            ara::core::Result<void> resultSetVoChnAttr = proxyAdapter->InitializeMethod<methods::SetVoChnAttr::Output>(methods::VoConfigServiceInterfaceSetVoChnAttrId);
            ThrowError(resultSetVoChnAttr);
            ara::core::Result<void> resultSetVoSource = proxyAdapter->InitializeMethod<methods::SetVoSource::Output>(methods::VoConfigServiceInterfaceSetVoSourceId);
            ThrowError(resultSetVoSource);
            ara::core::Result<void> resultGetVoSource = proxyAdapter->InitializeMethod<methods::GetVoSource::Output>(methods::VoConfigServiceInterfaceGetVoSourceId);
            ThrowError(resultGetVoSource);
            ara::core::Result<void> resultSetChnDisplayAttr = proxyAdapter->InitializeMethod<methods::SetChnDisplayAttr::Output>(methods::VoConfigServiceInterfaceSetChnDisplayAttrId);
            ThrowError(resultSetChnDisplayAttr);
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

    VoConfigServiceInterfaceProxy(const VoConfigServiceInterfaceProxy&) = delete;
    VoConfigServiceInterfaceProxy& operator=(const VoConfigServiceInterfaceProxy&) = delete;

    VoConfigServiceInterfaceProxy(VoConfigServiceInterfaceProxy&&) = default;
    VoConfigServiceInterfaceProxy& operator=(VoConfigServiceInterfaceProxy&&) = default;
    VoConfigServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          GetVoConfig(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoConfigId),
          GetVoChnAttr(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoChnAttrId),
          SetVoChnAttr(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetVoChnAttrId),
          SetVoSource(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetVoSourceId),
          GetVoSource(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoSourceId),
          SetChnDisplayAttr(proxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetChnDisplayAttrId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::vo::VoConfigServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::GetVoConfig GetVoConfig(preProxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetVoConfig::Output>(methods::VoConfigServiceInterfaceGetVoConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetVoChnAttr GetVoChnAttr(preProxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoChnAttrId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetVoChnAttr::Output>(methods::VoConfigServiceInterfaceGetVoChnAttrId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SetVoChnAttr SetVoChnAttr(preProxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetVoChnAttrId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetVoChnAttr::Output>(methods::VoConfigServiceInterfaceSetVoChnAttrId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SetVoSource SetVoSource(preProxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetVoSourceId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetVoSource::Output>(methods::VoConfigServiceInterfaceSetVoSourceId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetVoSource GetVoSource(preProxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceGetVoSourceId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetVoSource::Output>(methods::VoConfigServiceInterfaceGetVoSourceId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SetChnDisplayAttr SetChnDisplayAttr(preProxyAdapter->GetProxy(), methods::VoConfigServiceInterfaceSetChnDisplayAttrId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetChnDisplayAttr::Output>(methods::VoConfigServiceInterfaceSetChnDisplayAttrId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::vo::VoConfigServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::vo::VoConfigServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::vo::VoConfigServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::vo::VoConfigServiceInterface::ServiceIdentifier, specifier);
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
    methods::GetVoConfig GetVoConfig;
    methods::GetVoChnAttr GetVoChnAttr;
    methods::SetVoChnAttr SetVoChnAttr;
    methods::SetVoSource SetVoSource;
    methods::GetVoSource GetVoSource;
    methods::SetChnDisplayAttr SetChnDisplayAttr;
};
} // namespace proxy
} // namespace vo
} // namespace mdc

#endif // MDC_VO_VOCONFIGSERVICEINTERFACE_PROXY_H
