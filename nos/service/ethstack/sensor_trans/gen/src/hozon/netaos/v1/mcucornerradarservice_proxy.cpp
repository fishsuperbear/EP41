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
 * @file mcucornerradarservice_proxy.cpp
 * @brief proxy.cpp
 *
 */


#include "hozon/netaos/v1/mcucornerradarservice_proxy.h"
#include "ara/com/internal/proxy.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig mcucornerradarservice_1_0_manifest_config;

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
namespace McuCornerRadarService {
AlgCornerRadarTrackFR::AlgCornerRadarTrackFR(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgCornerRadarTrackFR::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgCornerRadarTrackFR::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgCornerRadarTrackFR::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgCornerRadarTrackFR::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgCornerRadarTrackFR::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgCornerRadarTrackFR::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgCornerRadarTrackFR::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgCornerRadarTrackFR::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgCornerRadarTrackFR::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgCornerRadarTrackArrayFrame>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgCornerRadarTrackFL::AlgCornerRadarTrackFL(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgCornerRadarTrackFL::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgCornerRadarTrackFL::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgCornerRadarTrackFL::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgCornerRadarTrackFL::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgCornerRadarTrackFL::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgCornerRadarTrackFL::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgCornerRadarTrackFL::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgCornerRadarTrackFL::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgCornerRadarTrackFL::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgCornerRadarTrackArrayFrame>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgCornerRadarTrackRR::AlgCornerRadarTrackRR(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgCornerRadarTrackRR::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgCornerRadarTrackRR::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgCornerRadarTrackRR::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgCornerRadarTrackRR::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgCornerRadarTrackRR::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgCornerRadarTrackRR::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgCornerRadarTrackRR::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgCornerRadarTrackRR::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgCornerRadarTrackRR::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgCornerRadarTrackArrayFrame>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgCornerRadarTrackRL::AlgCornerRadarTrackRL(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgCornerRadarTrackRL::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgCornerRadarTrackRL::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgCornerRadarTrackRL::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgCornerRadarTrackRL::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgCornerRadarTrackRL::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgCornerRadarTrackRL::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgCornerRadarTrackRL::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgCornerRadarTrackRL::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgCornerRadarTrackRL::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgCornerRadarTrackArrayFrame>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}

} // namespace McuCornerRadarService
} // namespace events


McuCornerRadarServiceProxy::McuCornerRadarServiceProxy(const ara::com::HandleType& handle_type)
: instance_(std::make_shared<ara::com::runtime::ProxyInstance>("mcucornerradarservice_1_0"))
, AlgCornerRadarTrackFR("AlgCornerRadarTrackFR", instance_, instance_->eventIdx("AlgCornerRadarTrackFR"))
, AlgCornerRadarTrackFL("AlgCornerRadarTrackFL", instance_, instance_->eventIdx("AlgCornerRadarTrackFL"))
, AlgCornerRadarTrackRR("AlgCornerRadarTrackRR", instance_, instance_->eventIdx("AlgCornerRadarTrackRR"))
, AlgCornerRadarTrackRL("AlgCornerRadarTrackRL", instance_, instance_->eventIdx("AlgCornerRadarTrackRL"))
{
    instance_->updateHandleType(handle_type);
}


ara::com::ServiceHandleContainer<ara::com::HandleType> McuCornerRadarServiceProxy::FindService(ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcucornerradarservice_1_0", instance);
}

ara::com::ServiceHandleContainer<ara::com::HandleType> McuCornerRadarServiceProxy::FindService(ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcucornerradarservice_1_0", instance);
}

ara::com::FindServiceHandle McuCornerRadarServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcucornerradarservice_1_0", instance);
}

ara::com::FindServiceHandle McuCornerRadarServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcucornerradarservice_1_0", instance);
}

void McuCornerRadarServiceProxy::StopFindService(ara::com::FindServiceHandle handle)
{
    ara::com::runtime::ProxyInstance::StopFindService(handle, "mcucornerradarservice_1_0");
}

ara::com::HandleType McuCornerRadarServiceProxy::GetHandle() const
{
    return instance_->getHandle();;
}

static ara::core::Result<void> McuCornerRadarServiceProxyInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("mcucornerradarservice_1_0", &mcucornerradarservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(McuCornerRadarServiceProxyInitialize);
}
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */