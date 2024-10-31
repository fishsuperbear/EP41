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
 * @file socpowerservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_SOCPOWERSERVICE_SKELETON_H_
#define HOZON_NETAOS_V1_SOCPOWERSERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "socpowerservice_common.h"

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{

namespace events{
namespace SocPowerService{
class SocSystemState : public ara::com::SkeletonMemberBase {
    public:
        SocSystemState(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::SocSysState& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::SocSysState> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::SocSysState> Allocate();
};
} //namespace SocPowerService
} //namespace events

class SocPowerServiceSkeleton;


class SocPowerServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        SocPowerServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        SocPowerServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~SocPowerServiceSkeleton();

        SocPowerServiceSkeleton(const SocPowerServiceSkeleton &) = delete;
        SocPowerServiceSkeleton &operator=(const SocPowerServiceSkeleton &) = delete;
        SocPowerServiceSkeleton(SocPowerServiceSkeleton &&) = default;
        SocPowerServiceSkeleton &operator=(SocPowerServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            events::SocPowerService::SocSystemState SocSystemState;

        private:
};
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_SOCPOWERSERVICE_SKELETON_H_
/* EOF */