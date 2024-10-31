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
 * @file mcufrontradarservice_proxy.cpp
 * @brief proxy.cpp
 *
 */


#include "hozon/netaos/v1/mcufrontradarservice_proxy.h"
#include "ara/com/internal/proxy.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig mcufrontradarservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace proxy{
static const ara::com::SomeipTransformationProps intra {
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
namespace McuFrontRadarService {
AlgFrontRadarTrack::AlgFrontRadarTrack(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgFrontRadarTrack::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgFrontRadarTrack::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgFrontRadarTrack::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgFrontRadarTrack::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgFrontRadarTrack::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgFrontRadarTrack::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgFrontRadarTrack::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgFrontRadarTrack::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgFrontRadarTrack::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgFrontRadarTrackArrayFrame>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}

} // namespace McuFrontRadarService
} // namespace events


McuFrontRadarServiceProxy::McuFrontRadarServiceProxy(const ara::com::HandleType& handle_type)
: instance_(std::make_shared<ara::com::runtime::ProxyInstance>("mcufrontradarservice_1_0"))
, AlgFrontRadarTrack("AlgFrontRadarTrack", instance_, instance_->eventIdx("AlgFrontRadarTrack"))
{
    instance_->updateHandleType(handle_type);
}


ara::com::ServiceHandleContainer<ara::com::HandleType> McuFrontRadarServiceProxy::FindService(ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcufrontradarservice_1_0", instance);
}

ara::com::ServiceHandleContainer<ara::com::HandleType> McuFrontRadarServiceProxy::FindService(ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcufrontradarservice_1_0", instance);
}

ara::com::FindServiceHandle McuFrontRadarServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcufrontradarservice_1_0", instance);
}

ara::com::FindServiceHandle McuFrontRadarServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcufrontradarservice_1_0", instance);
}

void McuFrontRadarServiceProxy::StopFindService(ara::com::FindServiceHandle handle)
{
    ara::com::runtime::ProxyInstance::StopFindService(handle, "mcufrontradarservice_1_0");
}

ara::com::HandleType McuFrontRadarServiceProxy::GetHandle() const
{
    return instance_->getHandle();;
}

static ara::core::Result<void> McuFrontRadarServiceProxyInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("mcufrontradarservice_1_0", &mcufrontradarservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(McuFrontRadarServiceProxyInitialize);
}
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */