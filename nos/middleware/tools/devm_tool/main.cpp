/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include <signal.h>

#include <iostream>
#include <string>
#include <vector>

#include <chrono>
#include <string.h>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

#include "get_read_did.h"
#include "get_ifdata.h"
#include "get_iostat.h"
// #include "get_cpu_info.h"
// #include "get_device_info.h"
// #include "get_upgrade.h"
#include "devm/include/devm_client.h"
#include "devm/include/devm_device_status.h"
#include "devm/include/devm_device_info.h"
#include "devm/include/devm_device_info_zmq.h"
#include "devm/include/devm_did_info.h"
#include "devm_tool_logger.h"
#include "get_upgrade_info.h"

using namespace hozon::netaos::tools;
using namespace hozon::netaos::devm;

// sig_atomic_t g_stopFlag = 0;
// void SigHandler(int signum) {
//     g_stopFlag = 1;
//     std::cout << "--- cfg SigHandler enter, signum [" << signum << "] ---" << std::endl;
// }
static void DevmToolPrintUsage()
{
    std::cout << "Usage: nos devm <options>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  read-did <did>,         Used to query did information, example 'nos devm read-did 0xf187'." << std::endl;
    std::cout << "  cpu-info,               Used to query cpu information." << std::endl;
    std::cout << "  dev-info,               Used to query devices information, Version, Temperature, Voltage." << std::endl;
    std::cout << "  dev-status,             Used to query devices status of soc, mcu, lidar, radar, imu, ussc." << std::endl;
    std::cout << "  upgrade <status/version/precheck/update/finish/result> [-f][all/orin/lidar/srrfl/srrfr/srrrl/srrrr] [filepath], Used to upgrade with a package." << std::endl;
    std::cout << "  help,                   Used to show the Usage." << std::endl;
    std::cout << std::endl;
}
int main(int argc, char *argv[])
{
    if(argc < 2) {
        std::cout << "Error command." << std::endl << std::endl;
        DevmToolPrintUsage();
        return -1;
    }
    DevmToolLogger::GetInstance().InitLogging();
    DevmToolLogger::GetInstance().CreateLogger("DEVMT");
    // signal(SIGINT, SigHandler);
    // signal(SIGTERM, SigHandler);

    std::vector<std::string> arguments;
    for (int i = 2; i < argc; ++i) {
        arguments.push_back(std::string(argv[i]));
    }

    std::string command;
    for (int i = 0; i < argc; ++i) {
        command += argv[i];
        command += " ";
    }
    DEVMTOOL_INFO << "command: "<< command;
    if (strcmp(argv[1], "read-did") == 0) {//nos devm read-did 0xf189/0xf187/..t;
        if (argc < 3) {
            std::cout << "Error command." << std::endl << std::endl;
            DevmToolPrintUsage();
            return -1;
        }

        std::map<uint16_t, std::string> tables= {
            {0x0110, "STR"}, {0x900F, "STR"}, {0xF170, "HEX"}, {0xF180, "STR"},
            {0xF186, "HEX"}, {0xF187, "STR"}, {0xF188, "STR"}, {0xF18A, "STR"},
            {0xF18B, "BCD"}, {0xF18C, "STR"}, {0xF190, "STR"}, {0xF191, "STR"},
            {0xF198, "STR"}, {0xF199, "BCD"}, {0xF19D, "BCD"}, {0xF1B0, "STR"},
            {0xF1BF, "STR"}, {0xF1C0, "STR"}, {0xF1D0, "STR"}, {0xF1E0, "STR"},
            {0xF1E1, "STR"}, {0xF1E2, "STR"}, {0xF1E3, "STR"}};
        DevmClientDidInfo readdid;
        std::string value = readdid.ReadDidInfo(argv[2]);

        uint16_t did{};
        try {
            did = std::stoi(argv[2], 0, 16);
        }
        catch (std::invalid_argument const &e) {
            std::cerr << "stoi error, " << e.what() << std::endl;
        }
        if (tables[did] == "STR") {
            printf("response did %s, value STR[%s]\n", argv[2], value.c_str());
        }
        else if (tables[did] == "HEX") {
            printf("response did %s, value HEX[", argv[2]);
            for (size_t i = 0; i < value.size(); i++) {
                printf("%02X ", (uint8_t)value.c_str()[i]);
            }
            if (value.size() > 0) {
                printf("\b");
            }
            printf("]\n");
        }
        else if (tables[did] == "BCD") {
            printf("response did %s, value BCD[", argv[2]);
            for (size_t i = 0; i < value.size(); i++) {
                printf("%02X ", (uint8_t)value.c_str()[i]);
            }
            if (value.size() > 0) {
                printf("\b");
            }
            printf("]\n");
        }
        else {
            printf("response did %s, value STR[%s]\n", argv[2], value.c_str());
        }

        char time_check[80];
        time_t rawtime;
        struct tm timeinfo;
        time(&rawtime);
        localtime_r(&rawtime, &timeinfo);
        strftime(time_check, sizeof(time_check), "%a %b %d %H:%M:%S %Y", &timeinfo);
        std::cout << time_check << std::endl;//time
    }
    else if (strcmp(argv[1], "ifdata") == 0) {//nos devm ifdata
        std::cout << "ifdata." << std::endl;
        IfDataInfo get_ifdata(arguments);
        get_ifdata.StartGetIfdata();
    }
    else if (strcmp(argv[1], "dev-info") == 0) {//nos devm dev-info
        // DevmClient* devmclient = new DevmClient();
        // devmclient->Init();
        // DeviceInfo devinfo = devmclient->GetDeviceInfo();
        // std::cout << "mcu_version: " << devinfo.mcu_version << std::endl;
        // devmclient->DeInit();
        // delete devmclient;


        // ZmqToolClient zmqtool{};
        // std::string value{};
        // zmqtool.Init();
        // zmqtool.DeviceInfo(value);
        // zmqtool.DeInit();
        // DeviceInfo devinfo{};
        // devinfo.mcu_version = value;


        DevmClientDeviceInfoZmq devmclient;
        DeviceInfo devinfo{};
        devmclient.SendRequestToServer(devinfo);
        TemperatureData temperature{};
        devmclient.GetTemperature(temperature);
        VoltageData voltage{};
        devmclient.GetVoltage(voltage);
        printf("%-20s%-20s\n", "Device", "Version");
        printf("%-20s%-20s\n", "----------", "----------");
        printf("%-20s%-20s\n", "SOC", devinfo.soc_version.length() == 0 ? "null" : devinfo.soc_version.c_str());
        printf("%-20s%-20s\n", "MCU", devinfo.mcu_version.length() == 0 ? "null" : devinfo.mcu_version.c_str());
        printf("%-20s%-20s\n", "DSV", devinfo.dsv_version.length() == 0 ? "null" : devinfo.dsv_version.c_str());
        printf("%-20s%-20s\n", "SWT", devinfo.swt_version.length() == 0 ? "null" : devinfo.swt_version.c_str());
        printf("%-20s%-20s\n", "USS", devinfo.uss_version.length() == 0 ? "null" : devinfo.uss_version.c_str());
        for (auto it: devinfo.sensor_version) {
            printf("%-20s%-20s\n", it.first.c_str(), it.second.length() == 0 ? "null" : it.second.c_str());
        }
        printf("\n");
        printf("%-20s%-20s\n", "Device", "Type");
        printf("%-20s%-20s\n", "----------", "----------");
        printf("%-20s%-20s\n", "soc", devinfo.soc_type.length() == 0 ? "null" : devinfo.soc_type.c_str());
        printf("%-20s%-20s\n", "mcu", devinfo.mcu_type.length() == 0 ? "null" : devinfo.mcu_type.c_str());
        printf("%-20s%-20s\n", "swt", devinfo.switch_type.length() == 0 ? "null" : devinfo.switch_type.c_str());
        printf("\n");

        printf("%-20s%-20s\n", "Device", "Temperature");
        printf("%-20s%-20s\n", "----------", "----------");
        printf("%-20s%-20.2f\n", "temp_soc", temperature.temp_soc);
        printf("%-20s%-20.2f\n", "temp_mcu", temperature.temp_mcu);
        printf("%-20s%-20.2f\n", "temp_ext0", temperature.temp_ext0);
        printf("%-20s%-20.2f\n", "temp_ext1", temperature.temp_ext1);
        printf("\n");
        printf("%-20s%-20s\n", "Device", "Voltage");
        printf("%-20s%-20s\n", "----------", "----------");
        printf("%-20s%-20s\n", "kl15",  voltage.kl15==0?"OFF":"ON");
        printf("%-20s%-20.2f\n", "kl30", voltage.kl30);
        printf("\n");
    }
    else if (strcmp(argv[1], "cpu-info") == 0) {//nos devm cpu-info
        // DevmClient* devmclient = new DevmClient();
        // devmclient->Init();
        // CpuData cpudata = devmclient->GetCpuInfo();
        // devmclient->DeInit();
        // delete devmclient;

        CpuData cpudata{};
        DevmClientCpuInfo devmclient;
        devmclient.SendRequestToServer(cpudata);

        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "-------------------- CPUINFO --------------------" << std::endl;
        std::cout << std::setw(24) << std::left << "Architecture:" << cpudata.architecture << std::endl;
        std::cout << std::setw(24) << std::left << "CPU(s):" << cpudata.cpus << std::endl;
        std::cout << std::setw(24) << std::left << "On-line CPU(s) list:" << cpudata.online_cpus << std::endl;
        std::cout << std::setw(24) << std::left << "Off-line CPU(s) list:" << cpudata.offline_cpus << std::endl;
        std::cout << std::setw(24) << std::left << "Model name:" << cpudata.model_name << std::endl;
        std::cout << std::setw(24) << std::left << "CPU max MHz:" << cpudata.cpu_max_mhz << std::endl;
        std::cout << std::setw(24) << std::left << "CPU min MHz:" << cpudata.cpu_min_mhz << std::endl;
        std::cout << std::setw(24) << std::left << "L1d cache:" << cpudata.l1d_catch << " KiB" << std::endl;
        std::cout << std::setw(24) << std::left << "L1i cache:" << cpudata.l1i_catch << " KiB" << std::endl;
        std::cout << std::setw(24) << std::left << "L2 cache:" << cpudata.l2_catch << " MiB" << std::endl;
        std::cout << std::setw(24) << std::left << "L3 cache:" << cpudata.l3_catch << " MiB" << std::endl;
        std::cout << std::setw(24) << std::left << "Temperature:" << cpudata.temp_cpu << "°C " << cpudata.temp_soc0 << "°C " << cpudata.temp_soc1 << "°C " << cpudata.temp_soc2 << "°C " << std::endl;
        std::cout << std::setw(24) << std::left << "Cpu usage:" << "[";
        for (auto vec : cpudata.cpus_usage) {
            std::cout << vec << "% ";
        }
        std::cout << "]" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "--------------- proc binding core ---------------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        for (auto pair : cpudata.cpu_binding) {
            std::cout << std::setw(24) << std::left << pair.first << ": " << pair.second << std::endl;
        }
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;

    }
    else if (strcmp(argv[1], "dev-status") == 0) {//nos devm dev-status
        // DevmClient* devmclient = new DevmClient();
        // devmclient->Init();
        // Devicestatus devstatus = devmclient->GetDeviceStatus();
        // devmclient->DeInit();
        // delete devmclient;

        DevmClientDeviceStatus devmclient;
        Devicestatus devstatus{};
        devmclient.SendRequestToServer(devstatus);
        printf("%-20s%-20s\n", "Device", "Status");
        printf("%-20s%-20s\n", "----------", "----------");
        printf("%-20s%-20s\n", "soc", devstatus.soc_status.size() == 0? "null":devstatus.soc_status.c_str());
        printf("%-20s%-20s\n", "mcu", devstatus.mcu_status.size() == 0? "null":devstatus.mcu_status.c_str());
        for (auto it: devstatus.camera_status) {
            printf("%-20s%-20s\n", it.first.c_str(), it.second.size() == 0? "null":it.second.c_str());
        }
        for (auto it: devstatus.lidar_status) {
            printf("%-20s%-20s\n", it.first.c_str(), it.second.size() == 0? "null":it.second.c_str());
        }
        for (auto it: devstatus.radar_status) {
            printf("%-20s%-20s\n", it.first.c_str(), it.second.size() == 0? "null":it.second.c_str());
        }
        for (auto it: devstatus.uss_status) {
            printf("%-20s%-20s\n", it.first.c_str(), it.second.size() == 0? "null":it.second.c_str());
        }
    }
    else if (strcmp(argv[1], "iostat") == 0) {//nos devm iostat
        IostatInfo get_ifdata(arguments);
        get_ifdata.StartGetIostat();
    }
    else if (strcmp(argv[1], "upgrade") == 0) {//nos devm upgrade status/precheck/..
        // UpgradeInfo get_upgrade_info(arguments);
        // get_upgrade_info.StartGetUpgradeInfo();
        UpgradeInfoZmq get_upgrade_info(arguments);
        get_upgrade_info.StartGetUpgradeInfo();
    }
    else if (strcmp(argv[1], "help") == 0 || strcmp(argv[1], "-h") == 0) {
        DevmToolPrintUsage();
    }
    else {
        std::cout << "Error command." << std::endl << std::endl;
        DevmToolPrintUsage();
    }

    return 0;
}

