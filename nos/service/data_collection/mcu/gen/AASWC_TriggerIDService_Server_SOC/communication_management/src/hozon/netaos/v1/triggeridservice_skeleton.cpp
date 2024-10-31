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
 * @file triggeridservice_skeleton.cpp
 * @brief skeleton.cpp
 *
 */

#include <utility>
using in_place_t = std::in_place_t;
#include "hozon/netaos/v1/triggeridservice_skeleton.h"
#include "ara/com/internal/skeleton.h"
#include "ara/com/internal/manifest_config.h"
STRUCTURE_REFLECTION_DEF(hozon::netaos::v1::skeleton::methods::TriggerIDService::MCUCloudTrigger::Output,Result);
extern ara::com::runtime::ComServiceManifestConfig triggeridservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{
static const ara::com::SomeipTransformationProps SomeipTransformationProps_TriggerIDService {
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
namespace methods{
namespace TriggerIDService{
MCUCloudTrigger::MCUCloudTrigger(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void MCUCloudTrigger::setCallback(Callback callback)
{
    instance_->setMethodAsyncRequestCallback(idx_,
            [this, callback](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result){
                std::uint8_t CloudTriggerID;
                executeMethodAsyncRequest<MCUCloudTrigger::Output>(instance_, idx_, payload, tag, e2e_result,
                                                                            callback,SomeipTransformationProps_TriggerIDService,CloudTriggerID);
            });
}
} // namespace TriggerIDService
} // namespace methods

TriggerIDServiceSkeleton::TriggerIDServiceSkeleton(ara::com::InstanceIdentifier instance, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("triggeridservice_1_0", instance, mode))
, mcucloudtrigger_("MCUCloudTrigger", instance_, instance_->methodIdx("MCUCloudTrigger"))
{
    mcucloudtrigger_.setCallback([this] (const std::uint8_t& CloudTriggerID) 
    ->ara::core::Future<methods::TriggerIDService::MCUCloudTrigger::Output>
    {
        return this->MCUCloudTrigger(CloudTriggerID);
    });
    instance_->start();
}

TriggerIDServiceSkeleton::TriggerIDServiceSkeleton(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("triggeridservice_1_0", instance_specifier, mode))
, mcucloudtrigger_("MCUCloudTrigger", instance_, instance_->methodIdx("MCUCloudTrigger"))
{
    mcucloudtrigger_.setCallback([this] (const std::uint8_t& CloudTriggerID) 
    ->ara::core::Future<methods::TriggerIDService::MCUCloudTrigger::Output>
    {
        return this->MCUCloudTrigger(CloudTriggerID);
    });
    instance_->start();
}

TriggerIDServiceSkeleton::~TriggerIDServiceSkeleton()
{
    StopOfferService();
}

void TriggerIDServiceSkeleton::OfferService()
{
    instance_->offerService();
}

void TriggerIDServiceSkeleton::StopOfferService()
{
    instance_->stopOfferService();
}

ara::core::Future<bool> TriggerIDServiceSkeleton::ProcessNextMethodCall()
{
    bool result = instance_->processNextMethodCall();
    ara::core::Promise<bool> promise;
    promise.set_value(result);
    return promise.get_future();
}

static ara::core::Result<void> TriggerIDServiceSkeletonInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("triggeridservice_1_0", &triggeridservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(TriggerIDServiceSkeletonInitialize);
}
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */