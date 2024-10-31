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
 * @file socdataservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_SOCDATASERVICE_SKELETON_H_
#define HOZON_NETAOS_V1_SOCDATASERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "socdataservice_common.h"

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{

namespace events{
namespace SocDataService{
class TrajData : public ara::com::SkeletonMemberBase {
    public:
        TrajData(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::HafEgoTrajectory& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafEgoTrajectory> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::HafEgoTrajectory> Allocate();
};
class PoseData : public ara::com::SkeletonMemberBase {
    public:
        PoseData(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::HafLocation& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafLocation> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::HafLocation> Allocate();
};
class SnsrFsnLaneDate : public ara::com::SkeletonMemberBase {
    public:
        SnsrFsnLaneDate(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::HafLaneDetectionOutArray& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafLaneDetectionOutArray> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::HafLaneDetectionOutArray> Allocate();
};
class SnsrFsnObj : public ara::com::SkeletonMemberBase {
    public:
        SnsrFsnObj(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::HafFusionOutArray& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::HafFusionOutArray> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::HafFusionOutArray> Allocate();
};
class ApaStateMachine : public ara::com::SkeletonMemberBase {
    public:
        ApaStateMachine(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::APAStateMachineFrame& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::APAStateMachineFrame> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::APAStateMachineFrame> Allocate();
};
class AlgEgoToMCU : public ara::com::SkeletonMemberBase {
    public:
        AlgEgoToMCU(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::AlgEgoToMcuFrame& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoToMcuFrame> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoToMcuFrame> Allocate();
};
class APAToMCUChassis : public ara::com::SkeletonMemberBase {
    public:
        APAToMCUChassis(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::AlgCanFdMsgFrame& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::AlgCanFdMsgFrame> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::AlgCanFdMsgFrame> Allocate();
};
class EgoToMCUChassis : public ara::com::SkeletonMemberBase {
    public:
        EgoToMCUChassis(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::AlgEgoHmiFrame& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoHmiFrame> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::AlgEgoHmiFrame> Allocate();
};
} //namespace SocDataService
} //namespace events

class SocDataServiceSkeleton;


class SocDataServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        SocDataServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        SocDataServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~SocDataServiceSkeleton();

        SocDataServiceSkeleton(const SocDataServiceSkeleton &) = delete;
        SocDataServiceSkeleton &operator=(const SocDataServiceSkeleton &) = delete;
        SocDataServiceSkeleton(SocDataServiceSkeleton &&) = default;
        SocDataServiceSkeleton &operator=(SocDataServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            events::SocDataService::TrajData TrajData;
            events::SocDataService::PoseData PoseData;
            events::SocDataService::SnsrFsnLaneDate SnsrFsnLaneDate;
            events::SocDataService::SnsrFsnObj SnsrFsnObj;
            events::SocDataService::ApaStateMachine ApaStateMachine;
            events::SocDataService::AlgEgoToMCU AlgEgoToMCU;
            events::SocDataService::APAToMCUChassis APAToMCUChassis;
            events::SocDataService::EgoToMCUChassis EgoToMCUChassis;

        private:
};
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_SOCDATASERVICE_SKELETON_H_
/* EOF */