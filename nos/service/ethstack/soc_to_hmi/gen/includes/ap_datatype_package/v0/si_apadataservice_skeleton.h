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
 * @file si_apadataservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_V0_SI_APADATASERVICE_SKELETON_H_
#define AP_DATATYPE_PACKAGE_V0_SI_APADATASERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "si_apadataservice_common.h"

namespace ap_datatype_package {
namespace v0 {
inline namespace v0 {
namespace skeleton{
namespace fields{
namespace SI_APAdataService{

class APAdataProperties_Field : public ara::com::SkeletonMemberBase {
    public:
        APAdataProperties_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-notifier
        void Update(const ::ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct& value);
        // has-getter
        void RegisterGetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct>()> handler);
};
class HPPInfo_Field : public ara::com::SkeletonMemberBase {
    public:
        HPPInfo_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-setter
        void RegisterSetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_HPP_Info_Struct>(const ::ap_datatype_package::datatypes::IDT_HPP_Info_Struct&)> handler);
};
class HPPLocationProperties_Field : public ara::com::SkeletonMemberBase {
    public:
        HPPLocationProperties_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-notifier
        void Update(const ::ap_datatype_package::datatypes::IDT_HPP_Location_Struct& value);
};
class HPPMapObjectDisplay_Field : public ara::com::SkeletonMemberBase {
    public:
        HPPMapObjectDisplay_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-notifier
        void Update(const ::ap_datatype_package::datatypes::IDT_HPP_MapObjectDisplay_struct& value);
};
class HPPdataProperties_Field : public ara::com::SkeletonMemberBase {
    public:
        HPPdataProperties_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-notifier
        void Update(const ::ap_datatype_package::datatypes::IDT_HPP_Path_Struct& value);
        // has-getter
        void RegisterGetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_HPP_Path_Struct>()> handler);
};
class InsInfoProperties_Field : public ara::com::SkeletonMemberBase {
    public:
        InsInfoProperties_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-setter
        void RegisterSetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_Ins_Info_Struct>(const ::ap_datatype_package::datatypes::IDT_Ins_Info_Struct&)> handler);
};
class NNSInfoProperties_Field : public ara::com::SkeletonMemberBase {
    public:
        NNSInfoProperties_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-setter
        void RegisterSetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_NNS_Info_Struct>(const ::ap_datatype_package::datatypes::IDT_NNS_Info_Struct&)> handler);
};
} //namespace SI_APAdataService
} //namespace fields


class SI_APAdataServiceSkeleton;


class SI_APAdataServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        SI_APAdataServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        SI_APAdataServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~SI_APAdataServiceSkeleton();

        SI_APAdataServiceSkeleton(const SI_APAdataServiceSkeleton &) = delete;
        SI_APAdataServiceSkeleton &operator=(const SI_APAdataServiceSkeleton &) = delete;
        SI_APAdataServiceSkeleton(SI_APAdataServiceSkeleton &&) = default;
        SI_APAdataServiceSkeleton &operator=(SI_APAdataServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            fields::SI_APAdataService::APAdataProperties_Field APAdataProperties_Field;
            fields::SI_APAdataService::HPPInfo_Field HPPInfo_Field;
            fields::SI_APAdataService::HPPLocationProperties_Field HPPLocationProperties_Field;
            fields::SI_APAdataService::HPPMapObjectDisplay_Field HPPMapObjectDisplay_Field;
            fields::SI_APAdataService::HPPdataProperties_Field HPPdataProperties_Field;
            fields::SI_APAdataService::InsInfoProperties_Field InsInfoProperties_Field;
            fields::SI_APAdataService::NNSInfoProperties_Field NNSInfoProperties_Field;

        private:
};
} // namespace skeleton
} // namespace v0
} // namespace v0
} // namespace ap_datatype_package


#endif // AP_DATATYPE_PACKAGE_V0_SI_APADATASERVICE_SKELETON_H_
/* EOF */