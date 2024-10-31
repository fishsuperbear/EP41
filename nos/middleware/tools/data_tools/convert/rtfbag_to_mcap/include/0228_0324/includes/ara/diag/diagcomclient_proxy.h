/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DIAG_DIAGCOMCLIENT_PROXY_H
#define ARA_DIAG_DIAGCOMCLIENT_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/diag/diagcomclient_common.h"
#include <string>

namespace ara {
namespace diag {
namespace proxy {
namespace events {
    using RequestTrigger = ara::com::internal::proxy::event::EventAdapter<::ara::diag::RequestTriggerType>;
    static constexpr ara::com::internal::EntityId DiagComClientRequestTriggerId = 52019U; //RequestTrigger_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId DiagComClientGetRequestId = 62768U; //GetRequest_method_hash
static constexpr ara::com::internal::EntityId DiagComClientSendResponseId = 17402U; //SendResponse_method_hash


class GetRequest {
public:
    using Output = ara::diag::methods::GetRequest::Output;

    GetRequest(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& spec, const ::UInt64& serialNumber)
    {
        return method_(spec, serialNumber);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::UInt64> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::UInt64> method_;
};

class SendResponse {
public:
    using Output = void;

    SendResponse(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<void> operator()(const ::String& spec, const ::UInt64& serialNumber, const ::Boolean& isPositive, const ::ara::diag::ByteVector& responseData)
    {
        return method_(spec, serialNumber, isPositive, responseData);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::UInt64, ::Boolean, ::ara::diag::ByteVector> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::UInt64, ::Boolean, ::ara::diag::ByteVector> method_;
};
} // namespace methods

class DiagComClientProxy {
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

    virtual ~DiagComClientProxy()
    {
        RequestTrigger.UnsetReceiveHandler();
        RequestTrigger.Unsubscribe();
    }

    explicit DiagComClientProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::ara::diag::DiagComClient::ServiceIdentifier, handle)),
          RequestTrigger(proxyAdapter->GetProxy(), events::DiagComClientRequestTriggerId, proxyAdapter->GetHandle(), ::ara::diag::DiagComClient::ServiceIdentifier),
          GetRequest(proxyAdapter->GetProxy(), methods::DiagComClientGetRequestId),
          SendResponse(proxyAdapter->GetProxy(), methods::DiagComClientSendResponseId){
            ara::core::Result<void> resultGetRequest = proxyAdapter->InitializeMethod<methods::GetRequest::Output>(methods::DiagComClientGetRequestId);
            ThrowError(resultGetRequest);
            ara::core::Result<void> resultSendResponse = proxyAdapter->InitializeMethod<methods::SendResponse::Output>(methods::DiagComClientSendResponseId);
            ThrowError(resultSendResponse);
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

    DiagComClientProxy(const DiagComClientProxy&) = delete;
    DiagComClientProxy& operator=(const DiagComClientProxy&) = delete;

    DiagComClientProxy(DiagComClientProxy&&) = default;
    DiagComClientProxy& operator=(DiagComClientProxy&&) = default;
    DiagComClientProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          RequestTrigger(proxyAdapter->GetProxy(), events::DiagComClientRequestTriggerId, proxyAdapter->GetHandle(), ::ara::diag::DiagComClient::ServiceIdentifier),
          GetRequest(proxyAdapter->GetProxy(), methods::DiagComClientGetRequestId),
          SendResponse(proxyAdapter->GetProxy(), methods::DiagComClientSendResponseId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::ara::diag::DiagComClient::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::GetRequest GetRequest(preProxyAdapter->GetProxy(), methods::DiagComClientGetRequestId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetRequest::Output>(methods::DiagComClientGetRequestId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SendResponse SendResponse(preProxyAdapter->GetProxy(), methods::DiagComClientSendResponseId);
            initResult = preProxyAdapter->InitializeMethod<methods::SendResponse::Output>(methods::DiagComClientSendResponseId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::diag::DiagComClient::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::diag::DiagComClient::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::diag::DiagComClient::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::diag::DiagComClient::ServiceIdentifier, specifier);
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
    events::RequestTrigger RequestTrigger;
    methods::GetRequest GetRequest;
    methods::SendResponse SendResponse;
};
} // namespace proxy
} // namespace diag
} // namespace ara

#endif // ARA_DIAG_DIAGCOMCLIENT_PROXY_H
