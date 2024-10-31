/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_define.h
 * Created on: Nov 21, 2023
 * Author: yanlongxiang
 *
 */
#pragma once
#include <string>
#include <vector>
#include <map>

namespace hozon {
namespace netaos {
namespace devm {


struct CpuData {
    std::string architecture;
    int cpus;
    std::string online_cpus;
    std::string offline_cpus;
    std::string model_name;
    float cpu_max_mhz;
    float cpu_min_mhz;
    int l1d_catch;
    int l1i_catch;
    int l2_catch;
    int l3_catch;
    float temp_cpu;
    float temp_soc0;
    float temp_soc1;
    float temp_soc2;
    std::map<std::string, std::string> cpu_binding;
    std::vector<float> cpus_usage;
};

struct DeviceInfo {
    std::string soc_version;
    std::string mcu_version;
    std::string swt_version;
    std::string dsv_version;
    std::string uss_version;
    std::string soc_type;
    std::string mcu_type;
    std::string switch_type;
    std::map<std::string, std::string> sensor_version;
};

struct Devicestatus {
    std::string soc_status;
    std::string mcu_status;
    std::map<std::string, std::string> camera_status;
    std::map<std::string, std::string> lidar_status;
    std::map<std::string, std::string> radar_status;
    std::map<std::string, std::string> imu_status;
    std::map<std::string, std::string> uss_status;
};


}  // namespace devm
}  // namespace netaos
}  // namespace hozon

