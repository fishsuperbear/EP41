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
 * @file mcufaultservice_proxy.h
 * @brief proxy.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_MCUFAULTSERVICE_PROXY_H_
#define HOZON_NETAOS_V1_MCUFAULTSERVICE_PROXY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include <memory>
#include "ara/core/instance_specifier.h"
#include "ara/com/types.h"
#include "mcufaultservice_common.h"


namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace proxy{
namespace methods{
namespace McuFaultService {
class FaultReport : public ara::com::ProxyMemberBase{
    public:
        struct Output{
            ::hozon::netaos::ResultEnum FaultReportResult;};

        FaultReport(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx);
        ara::core::Future<Output> operator() (const ::hozon::netaos::FaultDataStruct& FaultData);
};
class FaultToHMI : public ara::com::ProxyMemberBase{
    public:

        FaultToHMI(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx);
        void operator()(const ::hozon::netaos::signal& FaultToHmi);
};
} // namespace McuFaultService
} // namespace methods


class McuFaultServiceProxy{
    private:
        std::shared_ptr<ara::com::runtime::ProxyInstance> instance_;
    public:
        explicit McuFaultServiceProxy(const ara::com::HandleType& handle_type);
        ~McuFaultServiceProxy() = default;

        McuFaultServiceProxy(const McuFaultServiceProxy&) = delete;
        McuFaultServiceProxy& operator=(const McuFaultServiceProxy&) = delete;
        McuFaultServiceProxy(McuFaultServiceProxy&&) = default;
        McuFaultServiceProxy& operator=(McuFaultServiceProxy&&) = default;

        static ara::com::ServiceHandleContainer<ara::com::HandleType> FindService(ara::com::InstanceIdentifier instance = ara::com::InstanceIdentifier(ara::com::InstanceIdentifier::Any));

        static ara::com::ServiceHandleContainer<ara::com::HandleType> FindService(ara::core::InstanceSpecifier instance);

        static ara::com::FindServiceHandle StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler,
                                                            ara::com::InstanceIdentifier instance = ara::com::InstanceIdentifier(ara::com::InstanceIdentifier::Any));

        static ara::com::FindServiceHandle StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance);

        static void StopFindService(ara::com::FindServiceHandle handle);

        ara::com::HandleType GetHandle() const;

    public:
        methods::McuFaultService::FaultReport FaultReport;
        methods::McuFaultService::FaultToHMI FaultToHMI;
};
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_MCUFAULTSERVICE_PROXY_H_
/* EOF */