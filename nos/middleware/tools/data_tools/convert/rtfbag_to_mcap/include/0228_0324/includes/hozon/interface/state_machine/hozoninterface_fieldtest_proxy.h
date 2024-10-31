/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_FIELDTEST_PROXY_H
#define HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_FIELDTEST_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/state_machine/hozoninterface_fieldtest_common.h"
#include <string>

namespace hozon {
namespace interface {
namespace state_machine {
namespace proxy {
namespace events {
}

namespace fields {
    using hozonField = ara::com::internal::proxy::field::FieldAdapter<::hozon::statemachine::StateMachineFrame>;
    static constexpr ara::com::internal::EntityId HozonInterface_FieldTesthozonFieldId = 7370U; //hozonField_field_hash
    static constexpr ara::com::internal::EntityId HozonInterface_FieldTesthozonFieldSetterId = 11353U; //hozonField_setter_hash
    static constexpr ara::com::internal::EntityId HozonInterface_FieldTesthozonFieldGetterId = 1586U; //hozonField_getter_hash
}

namespace methods {

} // namespace methods

class HozonInterface_FieldTestProxy {
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

    virtual ~HozonInterface_FieldTestProxy()
    {
    }

    explicit HozonInterface_FieldTestProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, handle)),
          hozonField(proxyAdapter->GetProxy(), fields::HozonInterface_FieldTesthozonFieldId, proxyAdapter->GetHandle(), ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier) {
            hozonField.SetSetterEntityId(fields::HozonInterface_FieldTesthozonFieldSetterId);
            hozonField.SetGetterEntityId(fields::HozonInterface_FieldTesthozonFieldGetterId);
            ara::core::Result<void> resulthozonField = proxyAdapter->InitializeField<::hozon::statemachine::StateMachineFrame>(hozonField);
            ThrowError(resulthozonField);
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

    HozonInterface_FieldTestProxy(const HozonInterface_FieldTestProxy&) = delete;
    HozonInterface_FieldTestProxy& operator=(const HozonInterface_FieldTestProxy&) = delete;

    HozonInterface_FieldTestProxy(HozonInterface_FieldTestProxy&&) = default;
    HozonInterface_FieldTestProxy& operator=(HozonInterface_FieldTestProxy&&) = default;
    HozonInterface_FieldTestProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          hozonField(proxyAdapter->GetProxy(), fields::HozonInterface_FieldTesthozonFieldId, proxyAdapter->GetHandle(), ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier) {
        hozonField.SetSetterEntityId(fields::HozonInterface_FieldTesthozonFieldSetterId);
        hozonField.SetGetterEntityId(fields::HozonInterface_FieldTesthozonFieldGetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            fields::hozonField hozonField(preProxyAdapter->GetProxy(), fields::HozonInterface_FieldTesthozonFieldId, handle, ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier);
            hozonField.SetSetterEntityId(fields::HozonInterface_FieldTesthozonFieldSetterId);
            hozonField.SetGetterEntityId(fields::HozonInterface_FieldTesthozonFieldGetterId);
            initResult = preProxyAdapter->InitializeField<::hozon::statemachine::StateMachineFrame>(hozonField);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, specifier);
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
    fields::hozonField hozonField;
};
} // namespace proxy
} // namespace state_machine
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_FIELDTEST_PROXY_H
