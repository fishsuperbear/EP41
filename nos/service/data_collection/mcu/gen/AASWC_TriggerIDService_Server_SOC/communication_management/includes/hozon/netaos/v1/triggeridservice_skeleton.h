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
 * @file triggeridservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_TRIGGERIDSERVICE_SKELETON_H_
#define HOZON_NETAOS_V1_TRIGGERIDSERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "triggeridservice_common.h"

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{

class TriggerIDServiceSkeleton;

namespace methods{
namespace TriggerIDService{
class MCUCloudTrigger : public ara::com::SkeletonMemberBase {
    public:
        struct Output{
            std::uint8_t Result;};
    private:
        friend class hozon::netaos::v1::skeleton::TriggerIDServiceSkeleton;
        MCUCloudTrigger(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        using Callback = std::function<ara::core::Future<Output>(const std::uint8_t& CloudTriggerID)>;
        void setCallback(Callback callback);
};
} //namespace TriggerIDService
} //namespace methods


class TriggerIDServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        TriggerIDServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        TriggerIDServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~TriggerIDServiceSkeleton();

        TriggerIDServiceSkeleton(const TriggerIDServiceSkeleton &) = delete;
        TriggerIDServiceSkeleton &operator=(const TriggerIDServiceSkeleton &) = delete;
        TriggerIDServiceSkeleton(TriggerIDServiceSkeleton &&) = default;
        TriggerIDServiceSkeleton &operator=(TriggerIDServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            virtual ara::core::Future<methods::TriggerIDService::MCUCloudTrigger::Output> MCUCloudTrigger(const std::uint8_t& CloudTriggerID) = 0;

        private:
            skeleton::methods::TriggerIDService::MCUCloudTrigger mcucloudtrigger_;
};
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_TRIGGERIDSERVICE_SKELETON_H_
/* EOF */