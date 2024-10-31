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
 * @file si_adasdataservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_V0_SI_ADASDATASERVICE_SKELETON_H_
#define AP_DATATYPE_PACKAGE_V0_SI_ADASDATASERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "si_adasdataservice_common.h"

namespace ap_datatype_package {
namespace v0 {
inline namespace v0 {
namespace skeleton{
namespace fields{
namespace SI_ADASdataService{

class ADASdataProperties_Field : public ara::com::SkeletonMemberBase {
    public:
        ADASdataProperties_Field(const ara::core::String& name,const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        // has-notifier
        void Update(const ::ap_datatype_package::datatypes::IDT_ADAS_Dataproperties_Struct& value);
        // has-getter
        void RegisterGetHandler(std::function<ara::core::Future<::ap_datatype_package::datatypes::IDT_ADAS_Dataproperties_Struct>()> handler);
};
} //namespace SI_ADASdataService
} //namespace fields


class SI_ADASdataServiceSkeleton;


class SI_ADASdataServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        SI_ADASdataServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        SI_ADASdataServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~SI_ADASdataServiceSkeleton();

        SI_ADASdataServiceSkeleton(const SI_ADASdataServiceSkeleton &) = delete;
        SI_ADASdataServiceSkeleton &operator=(const SI_ADASdataServiceSkeleton &) = delete;
        SI_ADASdataServiceSkeleton(SI_ADASdataServiceSkeleton &&) = default;
        SI_ADASdataServiceSkeleton &operator=(SI_ADASdataServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            fields::SI_ADASdataService::ADASdataProperties_Field ADASdataProperties_Field;

        private:
};
} // namespace skeleton
} // namespace v0
} // namespace v0
} // namespace ap_datatype_package


#endif // AP_DATATYPE_PACKAGE_V0_SI_ADASDATASERVICE_SKELETON_H_
/* EOF */