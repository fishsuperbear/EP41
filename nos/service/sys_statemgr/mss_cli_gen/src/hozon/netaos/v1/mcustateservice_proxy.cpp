/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file mcustateservice_proxy.cpp
 * @brief proxy.cpp
 *
 */


#include "hozon/netaos/v1/mcustateservice_proxy.h"
#include "ara/com/internal/proxy.h"
#include "ara/com/internal/manifest_config.h"
STRUCTURE_REFLECTION_DEF(hozon::netaos::v1::proxy::methods::McuStateService::PowerModeReq::Output,RequestResult);
extern ara::com::runtime::ComServiceManifestConfig mcustateservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace proxy{
static const ara::com::SomeipTransformationProps props {
ara::com::DataAlignment::DataAlignment_8,                // alignment
ara::com::ByteOrderEnum::BYTE_ORDER_BIG_ENDIAN,                 // byteOrder
false,                  // implementsLegacyString
false,                  // isDynamicLengthFieldSize
true,                 // isSessionHandlingActive
ara::com::LengthFieldSize::LengthFieldSize_16,                   // sizeOfArrayLengthField
ara::com::LengthFieldSize::LengthFieldSize_8,                  // sizeOfStringLengthField
ara::com::LengthFieldSize::LengthFieldSize_32,                 // sizeOfStructLengthField
ara::com::LengthFieldSize::LengthFieldSize_32,                   // sizeOfUnionLengthField
ara::com::TypeSelectorFieldSize::TypeSelectorFieldSize_32,      // sizeOfUnionTypeSelectorField
ara::com::StringEncoding::UTF8     // stringEncoding
};

static const ara::com::SomeipTransformationProps kDefaultSomeipTransformationProps {
    ara::com::DataAlignment::DataAlignment_8,                  // alignment
    ara::com::ByteOrderEnum::BYTE_ORDER_BIG_ENDIAN,            // byteOrder
    false,                                                     // implementsLegacyString
    false,                                                     // isDynamicLengthFieldSize
    false,                                                     // isSessionHandlingActive
    ara::com::LengthFieldSize::LengthFieldSize_32,             // sizeOfArrayLengthField
    ara::com::LengthFieldSize::LengthFieldSize_32,             // sizeOfStringLengthField
    ara::com::LengthFieldSize::LengthFieldSize_0,             // sizeOfStructLengthField
    ara::com::LengthFieldSize::LengthFieldSize_32,             // sizeOfUnionLengthField
    ara::com::TypeSelectorFieldSize::TypeSelectorFieldSize_32,  // sizeOfUnionTypeSelectorField
    ara::com::StringEncoding::UTF8                             // stringEncoding
};
namespace events {
namespace McuStateService {
McuSystemState::McuSystemState(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void McuSystemState::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void McuSystemState::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState McuSystemState::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void McuSystemState::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void McuSystemState::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void McuSystemState::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void McuSystemState::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t McuSystemState::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> McuSystemState::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::McuSysState>(instance_, idx_, props, std::move(f), maxNumberOfSamples);
}

} // namespace McuStateService
} // namespace events

namespace methods{
namespace McuStateService{
PowerModeReq::PowerModeReq(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{
    setMethodResponseParameters<PowerModeReq::Output>(instance_, idx_,props);
}
ara::core::Future<PowerModeReq::Output>PowerModeReq::operator()(const ::hozon::netaos::PowerModeEnum& PowerMode)
{
    return sendMethodAsyncRequest<PowerModeReq::Output>(instance_, idx_,props , PowerMode);
}
} // namespace McuStateService
} // namespace methods


McuStateServiceProxy::McuStateServiceProxy(const ara::com::HandleType& handle_type)
: instance_(std::make_shared<ara::com::runtime::ProxyInstance>("mcustateservice_1_0"))
, McuSystemState("McuSystemState", instance_, instance_->eventIdx("McuSystemState"))
, PowerModeReq("PowerModeReq", instance_, instance_->methodIdx("PowerModeReq"))
{
    instance_->updateHandleType(handle_type);
}


ara::com::ServiceHandleContainer<ara::com::HandleType> McuStateServiceProxy::FindService(ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcustateservice_1_0", instance);
}

ara::com::ServiceHandleContainer<ara::com::HandleType> McuStateServiceProxy::FindService(ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcustateservice_1_0", instance);
}

ara::com::FindServiceHandle McuStateServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcustateservice_1_0", instance);
}

ara::com::FindServiceHandle McuStateServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcustateservice_1_0", instance);
}

void McuStateServiceProxy::StopFindService(ara::com::FindServiceHandle handle)
{
    ara::com::runtime::ProxyInstance::StopFindService(handle, "mcustateservice_1_0");
}

ara::com::HandleType McuStateServiceProxy::GetHandle() const
{
    return instance_->getHandle();;
}

static ara::core::Result<void> McuStateServiceProxyInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("mcustateservice_1_0", &mcustateservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(McuStateServiceProxyInitialize);
}
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */