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
 * @file vehiclecfgservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_VEHICLECFGSERVICE_SKELETON_H_
#define HOZON_NETAOS_V1_VEHICLECFGSERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "vehiclecfgservice_common.h"

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{

namespace events{
namespace VehicleCfgService{
class VehicleCfgUpdateEvent : public ara::com::SkeletonMemberBase {
    public:
        VehicleCfgUpdateEvent(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::VehicleCfgInfo& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::VehicleCfgInfo> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::VehicleCfgInfo> Allocate();
};
} //namespace VehicleCfgService
} //namespace events

class VehicleCfgServiceSkeleton;

namespace methods{
namespace VehicleCfgService{
class VehicleCfgUpdateRes : public ara::com::SkeletonMemberBase {
    public:
        struct Output{
            std::uint8_t Result;};
    private:
        friend class hozon::netaos::v1::skeleton::VehicleCfgServiceSkeleton;
        VehicleCfgUpdateRes(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        using Callback = std::function<ara::core::Future<Output>(const std::uint8_t& returnCode)>;
        void setCallback(Callback callback);
};
} //namespace VehicleCfgService
} //namespace methods


class VehicleCfgServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        VehicleCfgServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        VehicleCfgServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~VehicleCfgServiceSkeleton();

        VehicleCfgServiceSkeleton(const VehicleCfgServiceSkeleton &) = delete;
        VehicleCfgServiceSkeleton &operator=(const VehicleCfgServiceSkeleton &) = delete;
        VehicleCfgServiceSkeleton(VehicleCfgServiceSkeleton &&) = default;
        VehicleCfgServiceSkeleton &operator=(VehicleCfgServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            virtual ara::core::Future<methods::VehicleCfgService::VehicleCfgUpdateRes::Output> VehicleCfgUpdateRes(const std::uint8_t& returnCode) = 0;
            events::VehicleCfgService::VehicleCfgUpdateEvent VehicleCfgUpdateEvent;

        private:
            skeleton::methods::VehicleCfgService::VehicleCfgUpdateRes vehiclecfgupdateres_;
};
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_VEHICLECFGSERVICE_SKELETON_H_
/* EOF */