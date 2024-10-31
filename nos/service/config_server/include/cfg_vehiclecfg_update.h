

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 配置管理
 * Created on: Feb 7, 2023
 *
 */

#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_VEHICLECFG_UPDATE_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_VEHICLECFG_UPDATE_H_
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "cfg_data_def.h"
#include "hozon/netaos/impl_type_vehiclecfginfo.h"
#include "hozon/netaos/v1/vehiclecfgservice_skeleton.h"
#include "include/cfg_logger.h"

namespace hozon {
namespace netaos {
namespace cfg {
using namespace hozon::netaos::v1::v0::skeleton;
class CfgVehicleUpdate : public VehicleCfgServiceSkeleton {
 public:
    CfgVehicleUpdate();
    ~CfgVehicleUpdate();
    void Init();
    void DeInit();
    void vehicleCfgUpdateToMcu(std::vector<uint8_t> vehiclecfg);

 private:
    // method
    void vehicleCfgUpdateToMcu();
    CfgUpdateToMcuFlag notifyMcuUpdateFlag;
    virtual ara::core::Future<methods::VehicleCfgService::VehicleCfgUpdateRes::Output> VehicleCfgUpdateRes(const std::uint8_t& returnCode);
    const static uint8_t vehicleArrLen = 58;
    uint8_t destcfgdata[vehicleArrLen];
};

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_VEHICLECFG_UPDATE_H_
