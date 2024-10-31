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
 * @file mcucanmsgservice_proxy.cpp
 * @brief proxy.cpp
 *
 */


#include <utility>
using in_place_t = std::in_place_t;
#include "hozon/netaos/v1/mcucanmsgservice_proxy.h"
#include "ara/com/internal/proxy.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig mcucanmsgservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace proxy{
static const ara::com::SomeipTransformationProps SomeipTransformationProps_McuCANMsgService {
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
namespace McuCANMsgService {
CanMsgDrive::CanMsgDrive(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void CanMsgDrive::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void CanMsgDrive::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState CanMsgDrive::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void CanMsgDrive::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void CanMsgDrive::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void CanMsgDrive::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void CanMsgDrive::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t CanMsgDrive::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> CanMsgDrive::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::CanMsgDriveType>(instance_, idx_, SomeipTransformationProps_McuCANMsgService, std::move(f), maxNumberOfSamples);
}
CanMsgPark::CanMsgPark(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void CanMsgPark::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void CanMsgPark::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState CanMsgPark::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void CanMsgPark::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void CanMsgPark::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void CanMsgPark::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void CanMsgPark::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t CanMsgPark::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> CanMsgPark::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::CanMsgParkType>(instance_, idx_, SomeipTransformationProps_McuCANMsgService, std::move(f), maxNumberOfSamples);
}

} // namespace McuCANMsgService
} // namespace events


McuCANMsgServiceProxy::McuCANMsgServiceProxy(const ara::com::HandleType& handle_type)
: instance_(std::make_shared<ara::com::runtime::ProxyInstance>("mcucanmsgservice_1_0"))
, CanMsgDrive("CanMsgDrive", instance_, instance_->eventIdx("CanMsgDrive"))
, CanMsgPark("CanMsgPark", instance_, instance_->eventIdx("CanMsgPark"))
{
    instance_->updateHandleType(handle_type);
}


ara::com::ServiceHandleContainer<ara::com::HandleType> McuCANMsgServiceProxy::FindService(ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcucanmsgservice_1_0", instance);
}

ara::com::ServiceHandleContainer<ara::com::HandleType> McuCANMsgServiceProxy::FindService(ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcucanmsgservice_1_0", instance);
}

ara::com::FindServiceHandle McuCANMsgServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcucanmsgservice_1_0", instance);
}

ara::com::FindServiceHandle McuCANMsgServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcucanmsgservice_1_0", instance);
}

void McuCANMsgServiceProxy::StopFindService(ara::com::FindServiceHandle handle)
{
    ara::com::runtime::ProxyInstance::StopFindService(handle, "mcucanmsgservice_1_0");
}

ara::com::HandleType McuCANMsgServiceProxy::GetHandle() const
{
    return instance_->getHandle();;
}

static ara::core::Result<void> McuCANMsgServiceProxyInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("mcucanmsgservice_1_0", &mcucanmsgservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(McuCANMsgServiceProxyInitialize);
}
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */