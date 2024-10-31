/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: diag_server_uds_pub.h
 */

#ifndef DIAG_SERVER_UDS_PUB_H
#define DIAG_SERVER_UDS_PUB_H

#include <condition_variable>
#include <mutex>
#include <vector>
#include "ara/core/future.h"
#include "ara/core/promise.h"
#include "ara/core/instance_specifier.h"
#include "hozon/netaos/v1/socudsservice_skeleton.h"

using namespace hozon::netaos::v1::skeleton;

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsPub;
class DiagServerUdsImpl : public hozon::netaos::v1::skeleton::SoCUdsServiceSkeleton {
public:
    using Skeleton = hozon::netaos::v1::skeleton::SoCUdsServiceSkeleton;

     DiagServerUdsImpl(ara::com::InstanceIdentifier instanceID, ara::com::MethodCallProcessingMode mode, DiagServerUdsPub* handle);

    DiagServerUdsImpl(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode, DiagServerUdsPub* handle);

    virtual ~DiagServerUdsImpl();

    virtual ara::core::Future<methods::SoCUdsService::McuUdsRes::Output> McuUdsRes(const ::hozon::netaos::McuDiagDataType& McuDiagData);
private:
    // DiagServerUdsImpl();
    DiagServerUdsImpl(const DiagServerUdsImpl&);
    DiagServerUdsImpl& operator=(const DiagServerUdsImpl&);
    DiagServerUdsPub* handle_;
};

class DiagServerUdsPub {
public:
    static DiagServerUdsPub* getInstance();

    void Init();
    void DeInit();

    bool GetMcuDidsInfo(uint16_t did, std::vector<uint8_t>& uds);
    bool SendUdsEvent(std::vector<uint8_t> uds);
    virtual bool OnMcuUdsRes(const std::vector<uint8_t>& uds);

private:
    DiagServerUdsPub();
    DiagServerUdsPub(const DiagServerUdsPub&);
    DiagServerUdsPub& operator=(const DiagServerUdsPub&);

    bool UdsDidsParse(const std::vector<uint8_t>& uds);

private:
    std::shared_ptr<DiagServerUdsImpl> skeleton_;
    std::map<uint16_t, std::shared_ptr<std::vector<uint8_t>>> map_mcu_did_info_;
    static DiagServerUdsPub* instance_;
    static std::mutex mtx_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_PUB_H
