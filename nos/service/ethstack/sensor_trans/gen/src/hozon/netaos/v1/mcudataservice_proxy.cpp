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
 * @file mcudataservice_proxy.cpp
 * @brief proxy.cpp
 *
 */


#include "hozon/netaos/v1/mcudataservice_proxy.h"
#include "ara/com/internal/proxy.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig mcudataservice_1_0_manifest_config;

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
namespace McuDataService {
MbdDebugData::MbdDebugData(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void MbdDebugData::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void MbdDebugData::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState MbdDebugData::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void MbdDebugData::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void MbdDebugData::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void MbdDebugData::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void MbdDebugData::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t MbdDebugData::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> MbdDebugData::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::HafMbdDebug>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgImuInsInfo::AlgImuInsInfo(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgImuInsInfo::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgImuInsInfo::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgImuInsInfo::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgImuInsInfo::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgImuInsInfo::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgImuInsInfo::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgImuInsInfo::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgImuInsInfo::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgImuInsInfo::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgImuInsInfo>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgGNSSPosInfo::AlgGNSSPosInfo(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgGNSSPosInfo::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgGNSSPosInfo::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgGNSSPosInfo::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgGNSSPosInfo::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgGNSSPosInfo::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgGNSSPosInfo::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgGNSSPosInfo::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgGNSSPosInfo::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgGNSSPosInfo::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgGnssInfo>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgChassisInfo::AlgChassisInfo(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgChassisInfo::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgChassisInfo::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgChassisInfo::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgChassisInfo::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgChassisInfo::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgChassisInfo::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgChassisInfo::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgChassisInfo::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgChassisInfo::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgChassisInfo>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgPNCControl::AlgPNCControl(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgPNCControl::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgPNCControl::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgPNCControl::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgPNCControl::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgPNCControl::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgPNCControl::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgPNCControl::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgPNCControl::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgPNCControl::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::PNCControlState>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgMcuToEgo::AlgMcuToEgo(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgMcuToEgo::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgMcuToEgo::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgMcuToEgo::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgMcuToEgo::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgMcuToEgo::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgMcuToEgo::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgMcuToEgo::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgMcuToEgo::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgMcuToEgo::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::AlgMcuToEgoFrame>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}
AlgUssRawdata::AlgUssRawdata(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
: ProxyMemberBase(name, instance, idx)
{}

void AlgUssRawdata::Subscribe(std::size_t maxSampleCount)
{
    instance_->subscribeEvent(idx_, maxSampleCount);
}

void AlgUssRawdata::Unsubscribe()
{
    instance_->unsubscribeEvent(idx_);
}

ara::com::SubscriptionState AlgUssRawdata::GetSubscriptionState() const
{
    return instance_->getEventSubscriptionState(idx_);
}

void AlgUssRawdata::SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler)
{
    instance_->setEventSubscriptionStateChangeHandler(idx_, handler);
}

void AlgUssRawdata::UnsetSubscriptionStateChangeHandler()
{
    instance_->unsetEventSubscriptionStateChangeHandler(idx_);
}

void AlgUssRawdata::SetReceiveHandler(ara::com::EventReceiveHandler handler)
{
    instance_->setEventReceiveHandler(idx_, handler);
}

void AlgUssRawdata::UnsetReceiveHandler()
{
    instance_->unsetEventReceiveHandler(idx_);
}

size_t AlgUssRawdata::GetFreeSampleCount() const noexcept
{
    return instance_->getEventFreeSampleCount(idx_);
}

ara::core::Result<size_t> AlgUssRawdata::GetNewSamples(F&& f, size_t maxNumberOfSamples)
{
    return getEventNewSamples<::hozon::netaos::UssRawDataSet>(instance_, idx_, intra, std::move(f), maxNumberOfSamples);
}

} // namespace McuDataService
} // namespace events


McuDataServiceProxy::McuDataServiceProxy(const ara::com::HandleType& handle_type)
: instance_(std::make_shared<ara::com::runtime::ProxyInstance>("mcudataservice_1_0"))
, MbdDebugData("MbdDebugData", instance_, instance_->eventIdx("MbdDebugData"))
, AlgImuInsInfo("AlgImuInsInfo", instance_, instance_->eventIdx("AlgImuInsInfo"))
, AlgGNSSPosInfo("AlgGNSSPosInfo", instance_, instance_->eventIdx("AlgGNSSPosInfo"))
, AlgChassisInfo("AlgChassisInfo", instance_, instance_->eventIdx("AlgChassisInfo"))
, AlgPNCControl("AlgPNCControl", instance_, instance_->eventIdx("AlgPNCControl"))
, AlgMcuToEgo("AlgMcuToEgo", instance_, instance_->eventIdx("AlgMcuToEgo"))
, AlgUssRawdata("AlgUssRawdata", instance_, instance_->eventIdx("AlgUssRawdata"))
{
    instance_->updateHandleType(handle_type);
}


ara::com::ServiceHandleContainer<ara::com::HandleType> McuDataServiceProxy::FindService(ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcudataservice_1_0", instance);
}

ara::com::ServiceHandleContainer<ara::com::HandleType> McuDataServiceProxy::FindService(ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::FindService("mcudataservice_1_0", instance);
}

ara::com::FindServiceHandle McuDataServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcudataservice_1_0", instance);
}

ara::com::FindServiceHandle McuDataServiceProxy::StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::com::InstanceIdentifier instance)
{
    return ara::com::runtime::ProxyInstance::StartFindService(handler, "mcudataservice_1_0", instance);
}

void McuDataServiceProxy::StopFindService(ara::com::FindServiceHandle handle)
{
    ara::com::runtime::ProxyInstance::StopFindService(handle, "mcudataservice_1_0");
}

ara::com::HandleType McuDataServiceProxy::GetHandle() const
{
    return instance_->getHandle();;
}

static ara::core::Result<void> McuDataServiceProxyInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("mcudataservice_1_0", &mcudataservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(McuDataServiceProxyInitialize);
}
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */