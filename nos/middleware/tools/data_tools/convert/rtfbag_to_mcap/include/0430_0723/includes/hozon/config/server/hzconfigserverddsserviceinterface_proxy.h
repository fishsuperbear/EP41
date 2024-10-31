/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_PROXY_H
#define HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/config/server/hzconfigserverddsserviceinterface_common.h"
#include <string>

namespace hozon {
namespace config {
namespace server {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceReadVehicleConfigId = 52086U; //ReadVehicleConfig_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceWriteVehicleConfigId = 29793U; //WriteVehicleConfig_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceReadVINConfigId = 27579U; //ReadVINConfig_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceWriteVINConfigId = 44604U; //WriteVINConfig_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceReadSNConfigId = 42610U; //ReadSNConfig_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceWriteSNConfigId = 29526U; //WriteSNConfig_method_hash


class ReadVehicleConfig {
public:
    using Output = hozon::config::server::methods::ReadVehicleConfig::Output;

    ReadVehicleConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class WriteVehicleConfig {
public:
    using Output = hozon::config::server::methods::WriteVehicleConfig::Output;

    WriteVehicleConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::hozon::config::server::struct_config_array& vehicleConfigInfo)
    {
        return method_(vehicleConfigInfo);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::config::server::struct_config_array> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::config::server::struct_config_array> method_;
};

class ReadVINConfig {
public:
    using Output = hozon::config::server::methods::ReadVINConfig::Output;

    ReadVINConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class WriteVINConfig {
public:
    using Output = hozon::config::server::methods::WriteVINConfig::Output;

    WriteVINConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& VIN)
    {
        return method_(VIN);
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

class ReadSNConfig {
public:
    using Output = hozon::config::server::methods::ReadSNConfig::Output;

    ReadSNConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class WriteSNConfig {
public:
    using Output = hozon::config::server::methods::WriteSNConfig::Output;

    WriteSNConfig(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& SN)
    {
        return method_(SN);
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
} // namespace methods

class HzConfigServerDdsServiceInterfaceProxy {
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

    virtual ~HzConfigServerDdsServiceInterfaceProxy()
    {
    }

    explicit HzConfigServerDdsServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, handle)),
          ReadVehicleConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId),
          WriteVehicleConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId),
          ReadVINConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId),
          WriteVINConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId),
          ReadSNConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId),
          WriteSNConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId){
            ara::core::Result<void> resultReadVehicleConfig = proxyAdapter->InitializeMethod<methods::ReadVehicleConfig::Output>(methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId);
            ThrowError(resultReadVehicleConfig);
            ara::core::Result<void> resultWriteVehicleConfig = proxyAdapter->InitializeMethod<methods::WriteVehicleConfig::Output>(methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId);
            ThrowError(resultWriteVehicleConfig);
            ara::core::Result<void> resultReadVINConfig = proxyAdapter->InitializeMethod<methods::ReadVINConfig::Output>(methods::HzConfigServerDdsServiceInterfaceReadVINConfigId);
            ThrowError(resultReadVINConfig);
            ara::core::Result<void> resultWriteVINConfig = proxyAdapter->InitializeMethod<methods::WriteVINConfig::Output>(methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId);
            ThrowError(resultWriteVINConfig);
            ara::core::Result<void> resultReadSNConfig = proxyAdapter->InitializeMethod<methods::ReadSNConfig::Output>(methods::HzConfigServerDdsServiceInterfaceReadSNConfigId);
            ThrowError(resultReadSNConfig);
            ara::core::Result<void> resultWriteSNConfig = proxyAdapter->InitializeMethod<methods::WriteSNConfig::Output>(methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId);
            ThrowError(resultWriteSNConfig);
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

    HzConfigServerDdsServiceInterfaceProxy(const HzConfigServerDdsServiceInterfaceProxy&) = delete;
    HzConfigServerDdsServiceInterfaceProxy& operator=(const HzConfigServerDdsServiceInterfaceProxy&) = delete;

    HzConfigServerDdsServiceInterfaceProxy(HzConfigServerDdsServiceInterfaceProxy&&) = default;
    HzConfigServerDdsServiceInterfaceProxy& operator=(HzConfigServerDdsServiceInterfaceProxy&&) = default;
    HzConfigServerDdsServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          ReadVehicleConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId),
          WriteVehicleConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId),
          ReadVINConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId),
          WriteVINConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId),
          ReadSNConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId),
          WriteSNConfig(proxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::ReadVehicleConfig ReadVehicleConfig(preProxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReadVehicleConfig::Output>(methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::WriteVehicleConfig WriteVehicleConfig(preProxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::WriteVehicleConfig::Output>(methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReadVINConfig ReadVINConfig(preProxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReadVINConfig::Output>(methods::HzConfigServerDdsServiceInterfaceReadVINConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::WriteVINConfig WriteVINConfig(preProxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::WriteVINConfig::Output>(methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ReadSNConfig ReadSNConfig(preProxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::ReadSNConfig::Output>(methods::HzConfigServerDdsServiceInterfaceReadSNConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::WriteSNConfig WriteSNConfig(preProxyAdapter->GetProxy(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId);
            initResult = preProxyAdapter->InitializeMethod<methods::WriteSNConfig::Output>(methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, specifier);
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
    methods::ReadVehicleConfig ReadVehicleConfig;
    methods::WriteVehicleConfig WriteVehicleConfig;
    methods::ReadVINConfig ReadVINConfig;
    methods::WriteVINConfig WriteVINConfig;
    methods::ReadSNConfig ReadSNConfig;
    methods::WriteSNConfig WriteSNConfig;
};
} // namespace proxy
} // namespace server
} // namespace config
} // namespace hozon

#endif // HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_PROXY_H
