/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: zmq devm server
 */
#include <zipper.h>

#include "devm_server_impl_zmq.h"
#include "devm_server_logger.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
#include "devm_data_gathering.h"
#include "cpu_info.h"
#include "device_info.h"
#include "function_statistics.h"
#include "cfg_data.hpp"

using namespace hozon::netaos::zmqipc;
namespace hozon {
namespace netaos {
namespace devm_server {

DevmServerImplZmq::DevmServerImplZmq()
: ZmqIpcServer()
{
}

int32_t
DevmServerImplZmq::Init()
{
    FunctionStatistics func("DevmServerImplZmq::Init, ");
    auto res = Start(compress_log_service_name);
    return res;
}

int32_t
DevmServerImplZmq::DeInit()
{
    FunctionStatistics func("DevmServerImplZmq::DeInit finish, ");
    DEVM_LOG_DEBUG << "DevmServerImplZmq::DeInit";
    auto res = Stop();
    return res;
}

int32_t
DevmServerImplZmq::Process(const std::string& request, std::string& reply)
{
    std::lock_guard<std::mutex> lck(mtx_);
    DEVM_LOG_DEBUG << "DevmServerImplZmq::Process";
    std::string data(request.begin(), request.end());

    DevmReq req{};
    req.ParseFromString(data);
    std::string type = req.req_type();

    DEVM_LOG_INFO << "DevmServerImplZmq::Process type is " << type;
    if (type == "did_info") {
        std::string dids = req.data_value();
        std::vector<uint8_t> vec_data{};
        uint16_t did{};
        try {
            did = std::stoi(dids, 0, 16);
        }
        catch (std::invalid_argument const &e) {
            DEVM_LOG_ERROR << "stoi error, " << e.what();
        }
        DevmDataGathering::GetInstance().GetValueWithDid(did, vec_data);
        std::string str_data(vec_data.begin(), vec_data.end());
        DEVM_LOG_INFO << "DevmServerImplZmq::Process did " << dids << " value " << str_data;

        DevmDidInfo didinfo{};
        didinfo.set_did(did);
        didinfo.set_data_value(str_data);
        reply = didinfo.SerializeAsString();
    }
    else if (type == "cpu_info") {
        DevmCpuInfo cpuinfo{};
        CpuInfo cpu_;
        CpuData cpu_info_ = cpu_.GetAllInfo();
        DEVM_LOG_INFO << "after collect!";
        cpuinfo.set_architecture(cpu_info_.architecture);
        cpuinfo.set_cpus(cpu_info_.cpus);
        cpuinfo.set_online_cpus(cpu_info_.online_cpus);
        cpuinfo.set_offline_cpus(cpu_info_.offline_cpus);
        cpuinfo.set_model_name(cpu_info_.model_name);
        cpuinfo.set_cpu_max_mhz(cpu_info_.cpu_max_mhz);
        cpuinfo.set_cpu_min_mhz(cpu_info_.cpu_min_mhz);
        cpuinfo.set_l1d_catch(cpu_info_.l1d_catch);
        cpuinfo.set_l1i_catch(cpu_info_.l1i_catch);
        cpuinfo.set_l2_catch(cpu_info_.l2_catch);
        cpuinfo.set_l3_catch(cpu_info_.l3_catch);
        cpuinfo.set_temp_cpu(cpu_info_.temp_cpu);
        cpuinfo.set_temp_soc0(cpu_info_.temp_soc0);
        cpuinfo.set_temp_soc1(cpu_info_.temp_soc1);
        cpuinfo.set_temp_soc2(cpu_info_.temp_soc2);
        for (const auto& pair : cpu_info_.cpu_binding) {
            cpuinfo.mutable_cpu_binding()->insert({pair.first, pair.second});
        }
        for (auto cpu_usage : cpu_info_.cpus_usage) {
            cpuinfo.add_cpus_usage(cpu_usage);
        }
        reply = cpuinfo.SerializeAsString();
    }
    else if (type == "device_info") {
        DevmDeviceInfo devinfo{};
        DeviceInfo devic_reply = DeviceInfomation::getInstance()->GetData();
        std::string sensor_ver{};

        for(const auto& pair : version_tables) {
            sensor_ver.clear();
            ConfigParam::Instance()->GetParam<std::string>(pair.first, sensor_ver);
            devinfo.mutable_sensor_version()->insert({pair.second, sensor_ver});
        }

        devinfo.set_soc_version(devic_reply.soc_version);
        devinfo.set_mcu_version(devic_reply.mcu_version);
        devinfo.set_swt_version(devic_reply.swt_version);
        devinfo.set_dsv_version(devic_reply.dsv_version);
        devinfo.set_uss_version(devic_reply.uss_version);
        devinfo.set_soc_type(devic_reply.soc_type);
        devinfo.set_mcu_type(devic_reply.mcu_type);
        devinfo.set_switch_type(devic_reply.switch_type);

        // for (const auto& pair : devic_reply.sensor_version) {
        //     devinfo.mutable_sensor_version()->insert({pair.first, pair.second});
        // }
        reply = devinfo.SerializeAsString();
    }
    else if (type == "device_status") {
        DevmDeviceStatus devstatus{};
        uint8_t status{};
        CfgResultCode res{};
        std::string str_status{};
        res = ConfigParam::Instance()->GetParam<std::string>("system/soc_status", str_status);
        if (res != CONFIG_OK) {
            str_status = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/system/system.json", "system/soc_status");
        }
        devstatus.set_soc_status(str_status);
        str_status.clear();
        res = ConfigParam::Instance()->GetParam<std::string>("system/mcu_status", str_status);
        if (res != CONFIG_OK) {
            str_status = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/system/system.json", "system/mcu_status");
        }
        devstatus.set_mcu_status(str_status);

        for(const auto& pair : camera_status_tables) {
            // 0x0-Unkown, 0x1-link locked, 0x2-link unlock
            status = 0;
            res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
            str_status = res != CONFIG_OK ? "Unkown"
                        : status == 0 ? "Unkown"
                        : status == 1 ? "Link_locked"
                        : "Link_unlock";
            devstatus.mutable_camera_status()->insert({pair.second, str_status});
        }

        for(const auto& pair : lidar_status_tables) {
            // 0x0-Unkown, 0x1-Working, 0x2-link unlock
            status = 0;
            res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
            str_status = res != CONFIG_OK ? "Unkown"
                        : status == 0 ? "Unkown"
                        : status == 1 ? "Working"
                        : "Not Working";
            devstatus.mutable_lidar_status()->insert({pair.second, str_status});
        }

        for(const auto& pair : radar_status_tables) {
            status = 0;
            res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
            str_status = res != CONFIG_OK ? "Unkown"
                        : status == 0 ? "Unkown"
                        : status == 1 ? "Working"
                        : "Not Working";
            devstatus.mutable_radar_status()->insert({pair.second, str_status});
        }

        for(const auto& pair : uss_status_tables) {
            status = 0;
            res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
            str_status = res != CONFIG_OK ? "Unkown"
                        : status == 0 ? "Unkown"
                        : status == 1 ? "Working"
                        : "Not Working";
            devstatus.mutable_uss_status()->insert({pair.second, str_status});
        }

        reply = devstatus.SerializeAsString();
    }
    else if (type == "devm_temperature") {
        DevmTemperature mcu_temp_vol{};
        TemperatureData temp_vol = TemperatureDataInfo::getInstance()->GetData();
        mcu_temp_vol.set_soc_temp(temp_vol.soc_temp);
        mcu_temp_vol.set_mcu_temp(temp_vol.mcu_temp);
        mcu_temp_vol.set_ext0_temp(temp_vol.ext0_temp);
        mcu_temp_vol.set_ext1_temp(temp_vol.ext1_temp);
        DEVM_LOG_INFO << "soc_temp " << temp_vol.soc_temp;
        DEVM_LOG_INFO << "mcu_temp " << temp_vol.mcu_temp;
        DEVM_LOG_INFO << "ext0_temp " << temp_vol.ext0_temp;
        DEVM_LOG_INFO << "ext1_temp " << temp_vol.ext1_temp;
        reply = mcu_temp_vol.SerializeAsString();
    }
    else if (type == "devm_voltage") {
        DevmVoltage mcu_temp_vol{};
        VoltageData temp_vol = VoltageDataInfo::getInstance()->GetData();
        DEVM_LOG_INFO << "kl15_vol " << temp_vol.kl15_vol << ", kl30_vol " << temp_vol.kl30_vol;
        mcu_temp_vol.set_kl15_vol(temp_vol.kl15_vol);
        mcu_temp_vol.set_kl30_vol(temp_vol.kl30_vol);
        reply = mcu_temp_vol.SerializeAsString();
    }
    else {
        ;
    }

    return 0;
}


}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
