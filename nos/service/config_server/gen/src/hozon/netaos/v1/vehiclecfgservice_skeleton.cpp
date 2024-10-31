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
 * @file vehiclecfgservice_skeleton.cpp
 * @brief skeleton.cpp
 *
 */


#include "hozon/netaos/v1/vehiclecfgservice_skeleton.h"
#include "ara/com/internal/skeleton.h"
#include "ara/com/internal/manifest_config.h"
STRUCTURE_REFLECTION_DEF(hozon::netaos::v1::skeleton::methods::VehicleCfgService::VehicleCfgUpdateRes::Output,Result);
extern ara::com::runtime::ComServiceManifestConfig vehiclecfgservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{
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
namespace VehicleCfgService {
VehicleCfgUpdateEvent::VehicleCfgUpdateEvent(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void VehicleCfgUpdateEvent::Send(const ::hozon::netaos::VehicleCfgInfo& data)
{
    sendEventNotify<::hozon::netaos::VehicleCfgInfo>(instance_, idx_, props, data);
}

void VehicleCfgUpdateEvent::Send(ara::com::SampleAllocateePtr<::hozon::netaos::VehicleCfgInfo> data)
{
    sendEventNotify<::hozon::netaos::VehicleCfgInfo>(instance_, idx_, props, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::VehicleCfgInfo> VehicleCfgUpdateEvent::Allocate()
{
    return std::make_unique<::hozon::netaos::VehicleCfgInfo>();
}
} // namespace VehicleCfgService
} // namespace events
namespace methods{
namespace VehicleCfgService{
VehicleCfgUpdateRes::VehicleCfgUpdateRes(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void VehicleCfgUpdateRes::setCallback(Callback callback)
{
    instance_->setMethodAsyncRequestCallback(idx_,
            [this, callback](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result){
                std::uint8_t returnCode;
                executeMethodAsyncRequest<VehicleCfgUpdateRes::Output>(instance_, idx_, payload, tag, e2e_result,
                                                                            callback,props,returnCode);
            });
}
} // namespace VehicleCfgService
} // namespace methods

VehicleCfgServiceSkeleton::VehicleCfgServiceSkeleton(ara::com::InstanceIdentifier instance, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("vehiclecfgservice_1_0", instance, mode))
, VehicleCfgUpdateEvent("VehicleCfgUpdateEvent", instance_, instance_->eventIdx("VehicleCfgUpdateEvent"))
, vehiclecfgupdateres_("VehicleCfgUpdateRes", instance_, instance_->methodIdx("VehicleCfgUpdateRes"))
{
    vehiclecfgupdateres_.setCallback([this] (const std::uint8_t& returnCode) 
    ->ara::core::Future<methods::VehicleCfgService::VehicleCfgUpdateRes::Output>
    {
        return this->VehicleCfgUpdateRes(returnCode);
    });
    instance_->start();
}

VehicleCfgServiceSkeleton::VehicleCfgServiceSkeleton(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("vehiclecfgservice_1_0", instance_specifier, mode))
, VehicleCfgUpdateEvent("VehicleCfgUpdateEvent", instance_, instance_->eventIdx("VehicleCfgUpdateEvent"))
, vehiclecfgupdateres_("VehicleCfgUpdateRes", instance_, instance_->methodIdx("VehicleCfgUpdateRes"))
{
    vehiclecfgupdateres_.setCallback([this] (const std::uint8_t& returnCode) 
    ->ara::core::Future<methods::VehicleCfgService::VehicleCfgUpdateRes::Output>
    {
        return this->VehicleCfgUpdateRes(returnCode);
    });
    instance_->start();
}

VehicleCfgServiceSkeleton::~VehicleCfgServiceSkeleton()
{
    StopOfferService();
}

void VehicleCfgServiceSkeleton::OfferService()
{
    instance_->offerService();
}

void VehicleCfgServiceSkeleton::StopOfferService()
{
    instance_->stopOfferService();
}

ara::core::Future<bool> VehicleCfgServiceSkeleton::ProcessNextMethodCall()
{
    bool result = instance_->processNextMethodCall();
    ara::core::Promise<bool> promise;
    promise.set_value(result);
    return promise.get_future();
}

static ara::core::Result<void> VehicleCfgServiceSkeletonInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("vehiclecfgservice_1_0", &vehiclecfgservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(VehicleCfgServiceSkeletonInitialize);
}
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */