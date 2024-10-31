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
 * @file socfaultservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_SOCFAULTSERVICE_SKELETON_H_
#define HOZON_NETAOS_V1_SOCFAULTSERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "socfaultservice_common.h"

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{

class SocFaultServiceSkeleton;

namespace methods{
namespace SocFaultService{
class FaultEventReport : public ara::com::SkeletonMemberBase {
    public:
        struct Output{
            ::hozon::netaos::ResultEnum FaultRecvResult;};
    private:
        friend class hozon::netaos::v1::skeleton::SocFaultServiceSkeleton;
        FaultEventReport(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        using Callback = std::function<ara::core::Future<Output>(const ::hozon::netaos::FaultEvent& FaultRecvData)>;
        void setCallback(Callback callback);
};
} //namespace SocFaultService
} //namespace methods


class SocFaultServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        SocFaultServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        SocFaultServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~SocFaultServiceSkeleton();

        SocFaultServiceSkeleton(const SocFaultServiceSkeleton &) = delete;
        SocFaultServiceSkeleton &operator=(const SocFaultServiceSkeleton &) = delete;
        SocFaultServiceSkeleton(SocFaultServiceSkeleton &&) = default;
        SocFaultServiceSkeleton &operator=(SocFaultServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            virtual ara::core::Future<methods::SocFaultService::FaultEventReport::Output> FaultEventReport(const ::hozon::netaos::FaultEvent& FaultRecvData) = 0;

        private:
            skeleton::methods::SocFaultService::FaultEventReport faulteventreport_;
};
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_SOCFAULTSERVICE_SKELETON_H_
/* EOF */