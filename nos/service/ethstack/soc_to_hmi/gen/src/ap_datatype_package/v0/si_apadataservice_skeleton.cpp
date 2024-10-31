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
 * @file si_apadataservice_skeleton.cpp
 * @brief skeleton.cpp
 *
 */


#include "ap_datatype_package/v0/si_apadataservice_skeleton.h"
#include "ara/com/internal/skeleton.h"
#include "ara/com/internal/manifest_config.h"
extern ara::com::runtime::ComServiceManifestConfig si_apadataservice_0_0_manifest_config;

namespace ap_datatype_package {
namespace v0 {
inline namespace v0 {
namespace skeleton{
static const ara::com::SomeipTransformationProps APA_SOME_IP_Default_Transformer {
ara::com::DataAlignment::DataAlignment_32,                // alignment
ara::com::ByteOrderEnum::BYTE_ORDER_BIG_ENDIAN,                 // byteOrder
false,                  // implementsLegacyString
false,                  // isDynamicLengthFieldSize
false,                 // isSessionHandlingActive
ara::com::LengthFieldSize::LengthFieldSize_32,                   // sizeOfArrayLengthField
ara::com::LengthFieldSize::LengthFieldSize_32,                  // sizeOfStringLengthField
ara::com::LengthFieldSize::LengthFieldSize_0,                 // sizeOfStructLengthField
ara::com::LengthFieldSize::LengthFieldSize_0,                   // sizeOfUnionLengthField
ara::com::TypeSelectorFieldSize::TypeSelectorFieldSize_32,      // sizeOfUnionTypeSelectorField
ara::com::StringEncoding::UTF8     // stringEncoding
};
static const ara::com::SomeipTransformationProps AP_SOME_IP_Default_Transformer {
ara::com::DataAlignment::DataAlignment_32,                // alignment
ara::com::ByteOrderEnum::BYTE_ORDER_BIG_ENDIAN,                 // byteOrder
false,                  // implementsLegacyString
false,                  // isDynamicLengthFieldSize
false,                 // isSessionHandlingActive
ara::com::LengthFieldSize::LengthFieldSize_0,                   // sizeOfArrayLengthField
ara::com::LengthFieldSize::LengthFieldSize_32,                  // sizeOfStringLengthField
ara::com::LengthFieldSize::LengthFieldSize_0,                 // sizeOfStructLengthField
ara::com::LengthFieldSize::LengthFieldSize_0,                   // sizeOfUnionLengthField
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
namespace fields{
namespace SI_APAdataService{
APAdataProperties_Field::APAdataProperties_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-notifier
void APAdataProperties_Field::Update(const ::ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct& value)
{
    sendFieldNotify(instance_, idx_, APA_SOME_IP_Default_Transformer, value);
}
// has-getter
void APAdataProperties_Field::RegisterGetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct>()> handler)
{
       instance_->setFieldGetAsyncRequestCallback(idx_,
                [this, handler](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result){
                    executeFieldGetAsyncRequest<::ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct>(instance_, idx_, payload, tag, e2e_result, handler, APA_SOME_IP_Default_Transformer);
                });
}
HPPInfo_Field::HPPInfo_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-setter
void HPPInfo_Field::RegisterSetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_HPP_Info_Struct>(const ::ap_datatype_package::datatypes::IDT_HPP_Info_Struct&)> handler)
{
       instance_->setFieldSetAsyncRequestCallback(idx_,
                [this, handler](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result) {
                    executeFieldSetAsyncRequest<::ap_datatype_package::datatypes::IDT_HPP_Info_Struct>(instance_, idx_, payload, tag, e2e_result, handler, APA_SOME_IP_Default_Transformer);
                });
}
HPPLocationProperties_Field::HPPLocationProperties_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-notifier
void HPPLocationProperties_Field::Update(const ::ap_datatype_package::datatypes::IDT_HPP_Location_Struct& value)
{
    sendFieldNotify(instance_, idx_, APA_SOME_IP_Default_Transformer, value);
}
HPPMapObjectDisplay_Field::HPPMapObjectDisplay_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-notifier
void HPPMapObjectDisplay_Field::Update(const ::ap_datatype_package::datatypes::IDT_HPP_MapObjectDisplay_struct& value)
{
    sendFieldNotify(instance_, idx_, APA_SOME_IP_Default_Transformer, value);
}
HPPdataProperties_Field::HPPdataProperties_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-notifier
void HPPdataProperties_Field::Update(const ::ap_datatype_package::datatypes::IDT_HPP_Path_Struct& value)
{
    sendFieldNotify(instance_, idx_, APA_SOME_IP_Default_Transformer, value);
}
// has-getter
void HPPdataProperties_Field::RegisterGetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_HPP_Path_Struct>()> handler)
{
       instance_->setFieldGetAsyncRequestCallback(idx_,
                [this, handler](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result){
                    executeFieldGetAsyncRequest<::ap_datatype_package::datatypes::IDT_HPP_Path_Struct>(instance_, idx_, payload, tag, e2e_result, handler, APA_SOME_IP_Default_Transformer);
                });
}
InsInfoProperties_Field::InsInfoProperties_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-setter
void InsInfoProperties_Field::RegisterSetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_Ins_Info_Struct>(const ::ap_datatype_package::datatypes::IDT_Ins_Info_Struct&)> handler)
{
       instance_->setFieldSetAsyncRequestCallback(idx_,
                [this, handler](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result) {
                    executeFieldSetAsyncRequest<::ap_datatype_package::datatypes::IDT_Ins_Info_Struct>(instance_, idx_, payload, tag, e2e_result, handler, APA_SOME_IP_Default_Transformer);
                });
}
NNSInfoProperties_Field::NNSInfoProperties_Field(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
: SkeletonMemberBase(name, instance, idx)
{}
// has-setter
void NNSInfoProperties_Field::RegisterSetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_NNS_Info_Struct>(const ::ap_datatype_package::datatypes::IDT_NNS_Info_Struct&)> handler)
{
       instance_->setFieldSetAsyncRequestCallback(idx_,
                [this, handler](const std::shared_ptr<BufferList>& payload, std::shared_ptr<TagBase>& tag, int e2e_result) {
                    executeFieldSetAsyncRequest<::ap_datatype_package::datatypes::IDT_NNS_Info_Struct>(instance_, idx_, payload, tag, e2e_result, handler, APA_SOME_IP_Default_Transformer);
                });
}
} // namespace SI_APAdataService
} // namespace fields

SI_APAdataServiceSkeleton::SI_APAdataServiceSkeleton(ara::com::InstanceIdentifier instance, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("si_apadataservice_0_0", instance, mode))
, APAdataProperties_Field("APAdataProperties_Field", instance_, instance_->fieldIdx("APAdataProperties_Field"))
, HPPInfo_Field("HPPInfo_Field", instance_, instance_->fieldIdx("HPPInfo_Field"))
, HPPLocationProperties_Field("HPPLocationProperties_Field", instance_, instance_->fieldIdx("HPPLocationProperties_Field"))
, HPPMapObjectDisplay_Field("HPPMapObjectDisplay_Field", instance_, instance_->fieldIdx("HPPMapObjectDisplay_Field"))
, HPPdataProperties_Field("HPPdataProperties_Field", instance_, instance_->fieldIdx("HPPdataProperties_Field"))
, InsInfoProperties_Field("InsInfoProperties_Field", instance_, instance_->fieldIdx("InsInfoProperties_Field"))
, NNSInfoProperties_Field("NNSInfoProperties_Field", instance_, instance_->fieldIdx("NNSInfoProperties_Field"))
{
    instance_->start();
}

SI_APAdataServiceSkeleton::SI_APAdataServiceSkeleton(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode)
: instance_(std::make_shared<ara::com::runtime::SkeletonInstance>("si_apadataservice_0_0", instance_specifier, mode))
, APAdataProperties_Field("APAdataProperties_Field", instance_, instance_->fieldIdx("APAdataProperties_Field"))
, HPPInfo_Field("HPPInfo_Field", instance_, instance_->fieldIdx("HPPInfo_Field"))
, HPPLocationProperties_Field("HPPLocationProperties_Field", instance_, instance_->fieldIdx("HPPLocationProperties_Field"))
, HPPMapObjectDisplay_Field("HPPMapObjectDisplay_Field", instance_, instance_->fieldIdx("HPPMapObjectDisplay_Field"))
, HPPdataProperties_Field("HPPdataProperties_Field", instance_, instance_->fieldIdx("HPPdataProperties_Field"))
, InsInfoProperties_Field("InsInfoProperties_Field", instance_, instance_->fieldIdx("InsInfoProperties_Field"))
, NNSInfoProperties_Field("NNSInfoProperties_Field", instance_, instance_->fieldIdx("NNSInfoProperties_Field"))
{
    instance_->start();
}

SI_APAdataServiceSkeleton::~SI_APAdataServiceSkeleton()
{
    StopOfferService();
}

void SI_APAdataServiceSkeleton::OfferService()
{
    instance_->offerService();
}

void SI_APAdataServiceSkeleton::StopOfferService()
{
    instance_->stopOfferService();
}

ara::core::Future<bool> SI_APAdataServiceSkeleton::ProcessNextMethodCall()
{
    bool result = instance_->processNextMethodCall();
    ara::core::Promise<bool> promise;
    promise.set_value(result);
    return promise.get_future();
}

static ara::core::Result<void> SI_APAdataServiceSkeletonInitialize()
{
    ara::com::runtime::InstanceBase::loadManifest("si_apadataservice_0_0", &si_apadataservice_0_0_manifest_config);
    return ara::core::Result<void>::FromValue();
}

INITIALIZER(Initialize)
{
    ara::com::runtime::InstanceBase::registerInitializeFunction(SI_APAdataServiceSkeletonInitialize);
}
} // namespace skeleton
} // namespace v0
} // namespace v0
} // namespace ap_datatype_package

/* EOF */