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
 * @file socudsservice_skeleton.cpp
 * @brief skeleton.cpp
 *
 */


#include "hozon/netaos/v1/socudsservice_skeleton.h"
#include "ara/com/internal/skeleton.h"
#include "ara/com/internal/manifest_config.h"
STRUCTURE_REFLECTION_DEF(hozon::netaos::v1::skeleton::methods::SoCUdsService::McuUdsRes::Output,McuResult);
extern ara::com::runtime::ComServiceManifestConfig socudsservice_1_0_manifest_config;

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
namespace SoCUdsService {
SocUdsReq::SocUdsReq(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void SocUdsReq::Send(const ::hozon::netaos::SocUdsReqData& data)
{
    sendEventNotify<::hozon::netaos::SocUdsReqData>(instance_, idx_, props, data);
}

void SocUdsReq::Send(ara::com::SampleAllocateePtr<::hozon::netaos::SocUdsReqData> data)
{
    sendEventNotify<::hozon::netaos::SocUdsReqData>(instance_, idx_, props, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::SocUdsReqData> SocUdsReq::Allocate()
{
    return std::make_unique<::hozon::netaos::SocUdsReqData>();
}
} // namespace SoCUdsService
} // namespace events
namespace methods{
namespace SoCUdsService{
McuUdsRes::McuUdsRes(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void McuUdsRes::setCallback(Callback callback)
{
    instance_->setMethodAsyncRequestCallback(idx_,
            [this, callback](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result){
                ::hozon::netaos::McuDiagDataType McuDiagData;
                executeMethodAsyncRequest<McuUdsRes::Output>(instance_, idx_, payload, tag, e2e_result,
                                                                            callback,props,McuDiagData);
            });
}
} // namespace SoCUdsService
} // namespace methods

SoCUdsServiceSkeleton::SoCUdsServiceSkeleton(ara::com::InstanceIdentifier instance, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("socudsservice_1_0", instance, mode))
, SocUdsReq("SocUdsReq", instance_, instance_->eventIdx("SocUdsReq"))
, mcuudsres_("McuUdsRes", instance_, instance_->methodIdx("McuUdsRes"))
{
    mcuudsres_.setCallback([this] (const ::hozon::netaos::McuDiagDataType& McuDiagData) 
    ->ara::core::Future<methods::SoCUdsService::McuUdsRes::Output>
    {
        return this->McuUdsRes(McuDiagData);
    });
    instance_->start();
}

SoCUdsServiceSkeleton::SoCUdsServiceSkeleton(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("socudsservice_1_0", instance_specifier, mode))
, SocUdsReq("SocUdsReq", instance_, instance_->eventIdx("SocUdsReq"))
, mcuudsres_("McuUdsRes", instance_, instance_->methodIdx("McuUdsRes"))
{
    mcuudsres_.setCallback([this] (const ::hozon::netaos::McuDiagDataType& McuDiagData) 
    ->ara::core::Future<methods::SoCUdsService::McuUdsRes::Output>
    {
        return this->McuUdsRes(McuDiagData);
    });
    instance_->start();
}

SoCUdsServiceSkeleton::~SoCUdsServiceSkeleton()
{
    StopOfferService();
}

void SoCUdsServiceSkeleton::OfferService()
{
    instance_->offerService();
}

void SoCUdsServiceSkeleton::StopOfferService()
{
    instance_->stopOfferService();
}

ara::core::Future<bool> SoCUdsServiceSkeleton::ProcessNextMethodCall()
{
    bool result = instance_->processNextMethodCall();
    ara::core::Promise<bool> promise;
    promise.set_value(result);
    return promise.get_future();
}

static ara::core::Result<void> SoCUdsServiceSkeletonInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("socudsservice_1_0", &socudsservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(SoCUdsServiceSkeletonInitialize);
}
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */