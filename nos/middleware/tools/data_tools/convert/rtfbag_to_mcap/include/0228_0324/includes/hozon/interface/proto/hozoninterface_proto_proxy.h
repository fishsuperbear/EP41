/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_PROXY_H
#define HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/proto/hozoninterface_proto_common.h"
#include <string>

namespace hozon {
namespace interface {
namespace proto {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HozonInterface_ProtoProtoMethod_ReqWithRespId = 42085U; //ProtoMethod_ReqWithResp_method_hash


class ProtoMethod_ReqWithResp {
public:
    using Output = hozon::interface::proto::methods::ProtoMethod_ReqWithResp::Output;

    ProtoMethod_ReqWithResp(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& input)
    {
        return method_(input);
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

class HozonInterface_ProtoProxy {
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

    virtual ~HozonInterface_ProtoProxy()
    {
    }

    explicit HozonInterface_ProtoProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, handle)),
          ProtoMethod_ReqWithResp(proxyAdapter->GetProxy(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId){
            ara::core::Result<void> resultProtoMethod_ReqWithResp = proxyAdapter->InitializeMethod<methods::ProtoMethod_ReqWithResp::Output>(methods::HozonInterface_ProtoProtoMethod_ReqWithRespId);
            ThrowError(resultProtoMethod_ReqWithResp);
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

    HozonInterface_ProtoProxy(const HozonInterface_ProtoProxy&) = delete;
    HozonInterface_ProtoProxy& operator=(const HozonInterface_ProtoProxy&) = delete;

    HozonInterface_ProtoProxy(HozonInterface_ProtoProxy&&) = default;
    HozonInterface_ProtoProxy& operator=(HozonInterface_ProtoProxy&&) = default;
    HozonInterface_ProtoProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          ProtoMethod_ReqWithResp(proxyAdapter->GetProxy(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::ProtoMethod_ReqWithResp ProtoMethod_ReqWithResp(preProxyAdapter->GetProxy(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId);
            initResult = preProxyAdapter->InitializeMethod<methods::ProtoMethod_ReqWithResp::Output>(methods::HozonInterface_ProtoProtoMethod_ReqWithRespId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, specifier);
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
    methods::ProtoMethod_ReqWithResp ProtoMethod_ReqWithResp;
};
} // namespace proxy
} // namespace proto
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_PROXY_H
