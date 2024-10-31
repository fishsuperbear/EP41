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
 * @file socdataservice_skeleton.cpp
 * @brief skeleton.cpp
 *
 */


#include "hozon/netaos/v1/socdataservice_skeleton.h"
#include "ara/com/internal/skeleton.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig socdataservice_1_0_manifest_config;

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{
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
namespace SocDataService {
TrajData::TrajData(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void TrajData::Send(const ::hozon::netaos::HafEgoTrajectory& data)
{
    sendEventNotify<::hozon::netaos::HafEgoTrajectory>(instance_, idx_, intra, data);
}

void TrajData::Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafEgoTrajectory> data)
{
    sendEventNotify<::hozon::netaos::HafEgoTrajectory>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::HafEgoTrajectory> TrajData::Allocate()
{
    return std::make_unique<::hozon::netaos::HafEgoTrajectory>();
}
PoseData::PoseData(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void PoseData::Send(const ::hozon::netaos::HafLocation& data)
{
    sendEventNotify<::hozon::netaos::HafLocation>(instance_, idx_, intra, data);
}

void PoseData::Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafLocation> data)
{
    sendEventNotify<::hozon::netaos::HafLocation>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::HafLocation> PoseData::Allocate()
{
    return std::make_unique<::hozon::netaos::HafLocation>();
}
SnsrFsnLaneDate::SnsrFsnLaneDate(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void SnsrFsnLaneDate::Send(const ::hozon::netaos::HafLaneDetectionOutArray& data)
{
    sendEventNotify<::hozon::netaos::HafLaneDetectionOutArray>(instance_, idx_, intra, data);
}

void SnsrFsnLaneDate::Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafLaneDetectionOutArray> data)
{
    sendEventNotify<::hozon::netaos::HafLaneDetectionOutArray>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::HafLaneDetectionOutArray> SnsrFsnLaneDate::Allocate()
{
    return std::make_unique<::hozon::netaos::HafLaneDetectionOutArray>();
}
SnsrFsnObj::SnsrFsnObj(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void SnsrFsnObj::Send(const ::hozon::netaos::HafFusionOutArray& data)
{
    sendEventNotify<::hozon::netaos::HafFusionOutArray>(instance_, idx_, intra, data);
}

void SnsrFsnObj::Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafFusionOutArray> data)
{
    sendEventNotify<::hozon::netaos::HafFusionOutArray>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::HafFusionOutArray> SnsrFsnObj::Allocate()
{
    return std::make_unique<::hozon::netaos::HafFusionOutArray>();
}
ApaStateMachine::ApaStateMachine(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void ApaStateMachine::Send(const ::hozon::netaos::APAStateMachineFrame& data)
{
    sendEventNotify<::hozon::netaos::APAStateMachineFrame>(instance_, idx_, intra, data);
}

void ApaStateMachine::Send(ara::com::SampleAllocateePtr<::hozon::netaos::APAStateMachineFrame> data)
{
    sendEventNotify<::hozon::netaos::APAStateMachineFrame>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::APAStateMachineFrame> ApaStateMachine::Allocate()
{
    return std::make_unique<::hozon::netaos::APAStateMachineFrame>();
}
AlgEgoToMCU::AlgEgoToMCU(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void AlgEgoToMCU::Send(const ::hozon::netaos::AlgEgoToMcuFrame& data)
{
    sendEventNotify<::hozon::netaos::AlgEgoToMcuFrame>(instance_, idx_, intra, data);
}

void AlgEgoToMCU::Send(ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoToMcuFrame> data)
{
    sendEventNotify<::hozon::netaos::AlgEgoToMcuFrame>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoToMcuFrame> AlgEgoToMCU::Allocate()
{
    return std::make_unique<::hozon::netaos::AlgEgoToMcuFrame>();
}
APAToMCUChassis::APAToMCUChassis(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void APAToMCUChassis::Send(const ::hozon::netaos::AlgCanFdMsgFrame& data)
{
    sendEventNotify<::hozon::netaos::AlgCanFdMsgFrame>(instance_, idx_, intra, data);
}

void APAToMCUChassis::Send(ara::com::SampleAllocateePtr<::hozon::netaos::AlgCanFdMsgFrame> data)
{
    sendEventNotify<::hozon::netaos::AlgCanFdMsgFrame>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::AlgCanFdMsgFrame> APAToMCUChassis::Allocate()
{
    return std::make_unique<::hozon::netaos::AlgCanFdMsgFrame>();
}
EgoToMCUChassis::EgoToMCUChassis(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}

void EgoToMCUChassis::Send(const ::hozon::netaos::AlgEgoHmiFrame& data)
{
    sendEventNotify<::hozon::netaos::AlgEgoHmiFrame>(instance_, idx_, intra, data);
}

void EgoToMCUChassis::Send(ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoHmiFrame> data)
{
    sendEventNotify<::hozon::netaos::AlgEgoHmiFrame>(instance_, idx_, intra, (*data.get()));
}

ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoHmiFrame> EgoToMCUChassis::Allocate()
{
    return std::make_unique<::hozon::netaos::AlgEgoHmiFrame>();
}
} // namespace SocDataService
} // namespace events

SocDataServiceSkeleton::SocDataServiceSkeleton(ara::com::InstanceIdentifier instance, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("socdataservice_1_0", instance, mode))
, TrajData("TrajData", instance_, instance_->eventIdx("TrajData"))
, PoseData("PoseData", instance_, instance_->eventIdx("PoseData"))
, SnsrFsnLaneDate("SnsrFsnLaneDate", instance_, instance_->eventIdx("SnsrFsnLaneDate"))
, SnsrFsnObj("SnsrFsnObj", instance_, instance_->eventIdx("SnsrFsnObj"))
, ApaStateMachine("ApaStateMachine", instance_, instance_->eventIdx("ApaStateMachine"))
, AlgEgoToMCU("AlgEgoToMCU", instance_, instance_->eventIdx("AlgEgoToMCU"))
, APAToMCUChassis("APAToMCUChassis", instance_, instance_->eventIdx("APAToMCUChassis"))
, EgoToMCUChassis("EgoToMCUChassis", instance_, instance_->eventIdx("EgoToMCUChassis"))
{
    instance_->start();
}

SocDataServiceSkeleton::SocDataServiceSkeleton(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("socdataservice_1_0", instance_specifier, mode))
, TrajData("TrajData", instance_, instance_->eventIdx("TrajData"))
, PoseData("PoseData", instance_, instance_->eventIdx("PoseData"))
, SnsrFsnLaneDate("SnsrFsnLaneDate", instance_, instance_->eventIdx("SnsrFsnLaneDate"))
, SnsrFsnObj("SnsrFsnObj", instance_, instance_->eventIdx("SnsrFsnObj"))
, ApaStateMachine("ApaStateMachine", instance_, instance_->eventIdx("ApaStateMachine"))
, AlgEgoToMCU("AlgEgoToMCU", instance_, instance_->eventIdx("AlgEgoToMCU"))
, APAToMCUChassis("APAToMCUChassis", instance_, instance_->eventIdx("APAToMCUChassis"))
, EgoToMCUChassis("EgoToMCUChassis", instance_, instance_->eventIdx("EgoToMCUChassis"))
{
    instance_->start();
}

SocDataServiceSkeleton::~SocDataServiceSkeleton()
{
    StopOfferService();
}

void SocDataServiceSkeleton::OfferService()
{
    instance_->offerService();
}

void SocDataServiceSkeleton::StopOfferService()
{
    instance_->stopOfferService();
}

ara::core::Future<bool> SocDataServiceSkeleton::ProcessNextMethodCall()
{
    bool result = instance_->processNextMethodCall();
    ara::core::Promise<bool> promise;
    promise.set_value(result);
    return promise.get_future();
}

static ara::core::Result<void> SocDataServiceSkeletonInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("socdataservice_1_0", &socdataservice_1_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(SocDataServiceSkeletonInitialize);
}
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon

/* EOF */