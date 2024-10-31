/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CAM_CAMERA_CAMERAINTERNALDATAMETHODSERVICEINTERFACE_PROXY_H
#define MDC_CAM_CAMERA_CAMERAINTERNALDATAMETHODSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/cam/camera/camerainternaldatamethodserviceinterface_common.h"
#include <string>

namespace mdc {
namespace cam {
namespace camera {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId CameraInternalDataMethodServiceInterfaceGetCameraInternalDataId = 37965U; //GetCameraInternalData_method_hash


class GetCameraInternalData {
public:
    using Output = mdc::cam::camera::methods::GetCameraInternalData::Output;

    GetCameraInternalData(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::Int32& camId)
    {
        return method_(camId);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::Int32> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::Int32> method_;
};
} // namespace methods

class CameraInternalDataMethodServiceInterfaceProxy {
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

    virtual ~CameraInternalDataMethodServiceInterfaceProxy()
    {
    }

    explicit CameraInternalDataMethodServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::cam::camera::CameraInternalDataMethodServiceInterface::ServiceIdentifier, handle)),
          GetCameraInternalData(proxyAdapter->GetProxy(), methods::CameraInternalDataMethodServiceInterfaceGetCameraInternalDataId){
            ara::core::Result<void> resultGetCameraInternalData = proxyAdapter->InitializeMethod<methods::GetCameraInternalData::Output>(methods::CameraInternalDataMethodServiceInterfaceGetCameraInternalDataId);
            ThrowError(resultGetCameraInternalData);
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

    CameraInternalDataMethodServiceInterfaceProxy(const CameraInternalDataMethodServiceInterfaceProxy&) = delete;
    CameraInternalDataMethodServiceInterfaceProxy& operator=(const CameraInternalDataMethodServiceInterfaceProxy&) = delete;

    CameraInternalDataMethodServiceInterfaceProxy(CameraInternalDataMethodServiceInterfaceProxy&&) = default;
    CameraInternalDataMethodServiceInterfaceProxy& operator=(CameraInternalDataMethodServiceInterfaceProxy&&) = default;
    CameraInternalDataMethodServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          GetCameraInternalData(proxyAdapter->GetProxy(), methods::CameraInternalDataMethodServiceInterfaceGetCameraInternalDataId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::cam::camera::CameraInternalDataMethodServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::GetCameraInternalData GetCameraInternalData(preProxyAdapter->GetProxy(), methods::CameraInternalDataMethodServiceInterfaceGetCameraInternalDataId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetCameraInternalData::Output>(methods::CameraInternalDataMethodServiceInterfaceGetCameraInternalDataId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::cam::camera::CameraInternalDataMethodServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::cam::camera::CameraInternalDataMethodServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::cam::camera::CameraInternalDataMethodServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::cam::camera::CameraInternalDataMethodServiceInterface::ServiceIdentifier, specifier);
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
    methods::GetCameraInternalData GetCameraInternalData;
};
} // namespace proxy
} // namespace camera
} // namespace cam
} // namespace mdc

#endif // MDC_CAM_CAMERA_CAMERAINTERNALDATAMETHODSERVICEINTERFACE_PROXY_H
