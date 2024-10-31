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
 * @file socudsservice_skeleton.h
 * @brief skeleton.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_SOCUDSSERVICE_SKELETON_H_
#define HOZON_NETAOS_V1_SOCUDSSERVICE_SKELETON_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/types.h"
#include "ara/core/instance_specifier.h"
#include "socudsservice_common.h"

namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace skeleton{

namespace events{
namespace SoCUdsService{
class SocUdsReq : public ara::com::SkeletonMemberBase {
    public:
        SocUdsReq(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        void Send(const ::hozon::netaos::SocUdsReqData& data);
        void Send(ara::com::SampleAllocateePtr<::hozon::netaos::SocUdsReqData> data);
        ara::com::SampleAllocateePtr<::hozon::netaos::SocUdsReqData> Allocate();
};
} //namespace SoCUdsService
} //namespace events

class SoCUdsServiceSkeleton;

namespace methods{
namespace SoCUdsService{
class McuUdsRes : public ara::com::SkeletonMemberBase {
    public:
        struct Output{
            ::hozon::netaos::mcuResultEnum McuResult;};
    private:
        friend class hozon::netaos::v1::skeleton::SoCUdsServiceSkeleton;
        McuUdsRes(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx);
        using Callback = std::function<ara::core::Future<Output>(const ::hozon::netaos::McuDiagDataType& McuDiagData)>;
        void setCallback(Callback callback);
};
} //namespace SoCUdsService
} //namespace methods


class SoCUdsServiceSkeleton {
    private:
        std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    public:
        /// @uptrace{SWS_CM_00130}
        SoCUdsServiceSkeleton(ara::com::InstanceIdentifier instanceID,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        SoCUdsServiceSkeleton(ara::core::InstanceSpecifier instance_specifier,
                    ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent);

        virtual ~SoCUdsServiceSkeleton();

        SoCUdsServiceSkeleton(const SoCUdsServiceSkeleton &) = delete;
        SoCUdsServiceSkeleton &operator=(const SoCUdsServiceSkeleton &) = delete;
        SoCUdsServiceSkeleton(SoCUdsServiceSkeleton &&) = default;
        SoCUdsServiceSkeleton &operator=(SoCUdsServiceSkeleton &&) = default;

        void OfferService();

        void StopOfferService();

        ara::core::Future<bool> ProcessNextMethodCall();

        public:
            virtual ara::core::Future<methods::SoCUdsService::McuUdsRes::Output> McuUdsRes(const ::hozon::netaos::McuDiagDataType& McuDiagData) = 0;
            events::SoCUdsService::SocUdsReq SocUdsReq;

        private:
            skeleton::methods::SoCUdsService::McuUdsRes mcuudsres_;
};
} // namespace skeleton
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_SOCUDSSERVICE_SKELETON_H_
/* EOF */