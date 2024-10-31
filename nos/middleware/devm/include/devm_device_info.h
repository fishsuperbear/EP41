/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_device_info.cpp
 * Created on: Nov 21, 2023
 * Author: yanlongxiang
 *
 */
#pragma once

#include "devm_define.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"
#include "cfg/include/config_param.h"

using namespace hozon::netaos::zmqipc;
using namespace hozon::netaos::cfg;

namespace hozon {
namespace netaos {
namespace devm {


const uint8_t VIN_DATA_LENGTH = 17;
const uint8_t VEHICLE_CFG_WORD_DATA_LENGTH = 58;
const uint8_t ECU_SW_DATA_LENGTH = 8;
const uint8_t TESTER_SN_DATA_LENGTH = 10;
const uint8_t PROGRAMMING_DATE_DATA_LENGTH = 4;
const uint8_t ECU_TYPE_DATA_LENGTH = 8;
const uint8_t ECU_INSTALL_DATE_LENGTH = 4;
const uint8_t ESK_NUMBER_LENGTH = 16;
const uint8_t BOOT_SW_DATA_LENGTH = 8;
const uint8_t ECU_SOFTWARE_NUMBER_LENGTH = 8;
const uint8_t HARDWARE_NUM_LENGTH = 5;
const uint8_t ECU_PART_NUM_LENGTH = 13;
const uint8_t ECU_SERIAL_NUMBER_LENGTH = 18;

struct TemperatureData {
    float temp_soc = -1;
    float temp_mcu = -1;
    float temp_ext0 = -1;
    float temp_ext1 = -1;
};

struct VoltageData {
    int32_t kl15 = -1;
    float kl30 = -1;
};

class DevmClientDeviceInfo {
public:
    DevmClientDeviceInfo();
    ~DevmClientDeviceInfo();
    int32_t SendRequestToServer(DeviceInfo& resp);
    bool GetVinNumber(std::string& vin_number);
    bool GetEcuSerialNumber(std::string& sn_number);
    bool GetEcuParterNum(std::string& ecu_parter_num);
    bool GetVehicleCfg(std::string& ecu_vehicle_cfg);
    bool GetTemperature(struct TemperatureData& temp);
    bool GetVoltage(struct VoltageData& voltage);

private:
    std::shared_ptr<ZmqIpcClient> client_{};
    ConfigParam* cfg_mgr_;
};

}  // namespace devm
}  // namespace netaos
}  // namespace hozon

