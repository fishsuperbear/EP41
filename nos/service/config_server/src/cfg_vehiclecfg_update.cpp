
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: datamanager
 */

#include "include/cfg_vehiclecfg_update.h"

#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>

#include <ara/core/promise.h>
namespace hozon {
namespace netaos {
namespace cfg {

CfgVehicleUpdate::CfgVehicleUpdate() : VehicleCfgServiceSkeleton(ara::com::InstanceIdentifier("1"), ara::com::MethodCallProcessingMode::kEvent) {}
CfgVehicleUpdate::~CfgVehicleUpdate() {}

void CfgVehicleUpdate::Init() {
    CONFIG_LOG_INFO << "Init  ";
    notifyMcuUpdateFlag = CfgUpdateToMcuFlag::NOTIFY_PEDDING;
    this->OfferService();
    vehicleCfgUpdateToMcu();
}
void CfgVehicleUpdate::DeInit() {
    CONFIG_LOG_INFO << "DeInit  ";
    notifyMcuUpdateFlag = CfgUpdateToMcuFlag::NOTIFY_STOP;
    this->StopOfferService();
}

void CfgVehicleUpdate::vehicleCfgUpdateToMcu(std::vector<uint8_t> vehiclecfg) {
    uint32_t length = vehiclecfg.size();
    CONFIG_LOG_INFO << "vehicleCfgUpdateToMcu::datalengeth   " << length;
    if (length == vehicleArrLen) {
        std::copy(vehiclecfg.begin(), vehiclecfg.end(), destcfgdata);
        notifyMcuUpdateFlag = CfgUpdateToMcuFlag::NOTIFY_UPDATING;
    }
}

ara::core::Future<methods::VehicleCfgService::VehicleCfgUpdateRes::Output> CfgVehicleUpdate::VehicleCfgUpdateRes(const std::uint8_t& returnCode) {
    CONFIG_LOG_INFO << "VehicleCfgUpdateRes returnCode: " << returnCode;
    if (returnCode == 0) {
        notifyMcuUpdateFlag = CfgUpdateToMcuFlag::NOTIFY_PEDDING;
    }
    methods::VehicleCfgService::VehicleCfgUpdateRes::Output output;
    output.Result = returnCode;
    decltype(VehicleCfgServiceSkeleton::VehicleCfgUpdateRes(returnCode))::PromiseType promise;
    promise.set_value(std::move(output));
    return promise.get_future();
}

void CfgVehicleUpdate::vehicleCfgUpdateToMcu() {
    CONFIG_LOG_INFO << "begin...";
    std::thread sendThread([this]() -> void {
        pthread_setname_np(pthread_self(), "vehicleCfgUpdateToMcu");
        CONFIG_LOG_INFO << "sendThread begin...";
        uint32_t seqcount = 0;
        while (notifyMcuUpdateFlag) {
            struct timespec start_time;
            if (0 != clock_gettime(CLOCK_REALTIME, &start_time)) {
                CONFIG_LOG_INFO << "clock_gettime fail ";
            }
            if (seqcount % 10 == 0) {
                if (notifyMcuUpdateFlag == CfgUpdateToMcuFlag::NOTIFY_UPDATING) {
                    auto data = this->VehicleCfgUpdateEvent.Allocate();
                    if (data == nullptr) {
                        CONFIG_LOG_INFO << "VehicleCfgUpdateEvent Allocate failed: ";
                    } else {
                        std::copy(destcfgdata, destcfgdata + vehicleArrLen, data->begin());
                        this->VehicleCfgUpdateEvent.Send(std::move(data));
                        if (seqcount % 600 == 0) {
                            CONFIG_LOG_INFO << "VehicleCfgUpdateEvent::data Send count:" << seqcount / 10 << " datasize: " << data.get()->size();
                        }
                    }
                }
            }
            struct timespec end_time;
            if (0 != clock_gettime(CLOCK_REALTIME, &end_time)) {
                CONFIG_LOG_INFO << "clock_gettime fail ";
            }
            const uint32_t cycle_time = 100 * 1000 * 1000U;
            long long use_time = (end_time.tv_sec - start_time.tv_sec) * 1000 * 1000 * 1000 + (end_time.tv_nsec - start_time.tv_nsec);
            long long sleepTime = (cycle_time - use_time);
            if (sleepTime > 0) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(sleepTime));
            }
            seqcount++;
        }
        CONFIG_LOG_INFO << "sendThread end...";
    });
    sendThread.detach();
    CONFIG_LOG_INFO << "end...";
}

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
