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
 * @file socpowerservice_skeleton.cpp
 * @brief skeleton.cpp
 *
 */


#include "hozon/netaos/v1/socpowerservice_skeleton.h"
#include "ara/com/internal/skeleton.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig socpowerservice_1_0_manifest_config;

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
namespace SocPowerService {
SocSystemState::SocSystemState(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void SocSystemState::Send(const ::hozon::netaos::SocSysState& data)
{
    sendEventNotify<::hozon::netaos::SocSysState>(instance_, idx_, props, data);
}

void SocSystemState::Send(ara::com::SampleAllocateePtr<::hozon::netaos::SocSysState> data)
{
    sendEventNotify<::hozon::netaos::SocSysState>(instance_, idx_, props, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::SocSysState> SocSystemState::Allocate()
{
    return std::make_unique<::hozon::netaos::SocSysState>();
}
} // namespace SocPowerService
} // namespace events

SocPowerServiceSkeleton::SocPowerServiceSkeleton(ara::com::InstanceIdentifier instance, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("socpowerservice_1_0", instance, mode))
, SocSystemState("SocSystemState", instance_, instance_->eventIdx("SocSystemState"))
{
    instance_->start();
}

SocPowerServiceSkeleton::SocPowerServiceSkeleton(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("socpowerservice_1_0", instance_specifier, mode))
, SocSystemState("SocSystemState", instance_, instance_->eventIdx("SocSystemState"))
{
    instance_->start();
}

SocPowerServiceSkeleton::~SocPowerServiceSkeleton()
{
    StopOfferService();
}

void SocPowerServiceSkeleton::OfferService()
{
    instance_->offerService();
}

void SocPowerServiceSkeleton::StopOfferService()
{
    instance_->stopOfferService();
}

ara::core::Future<bool> SocPowerServiceSkeleton::ProcessNextMethodCall()
{
    bool result = instance_->processNextMethodCall();
    ara::core::Promise<bool> promise;
    promise.set_value(result);
    return promise.get_future();
}

static ara::core::Result<void> SocPowerServiceSkeletonInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("socpowerservice_1_0", &socpowerservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(SocPowerServiceSkeletonInitialize);
}
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */