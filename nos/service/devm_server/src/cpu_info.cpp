#include "cpu_info.h"
#include <stdio.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "devm_server_logger.h"
#include "nlohmann/json.hpp"

#define MAX_SIZE 1024
#define M_SIZE 512
#define S_SIZE 128

namespace hozon {
namespace netaos {
namespace devm_server {

static const std::string folderpath = "/app/runtime_service/";
static const std::string statpath = "/proc/stat";
static const std::string temppath0 = "/sys/devices/virtual/thermal/thermal_zone0/temp";
static const std::string temppath5 = "/sys/devices/virtual/thermal/thermal_zone5/temp";
static const std::string temppath6 = "/sys/devices/virtual/thermal/thermal_zone6/temp";
static const std::string temppath7 = "/sys/devices/virtual/thermal/thermal_zone7/temp";

static std::map<std::string, std::string> binding_map;

using json = nlohmann::json;

namespace fs = std::filesystem;
static std::string get_mpstat_output();
static std::string change_path_name(const std::string&);
static std::string get_lscpu_output();
static std::string get_field_value(const std::string& output__, const std::string& token__);
static std::string remove_prefix_space(const std::string& str);
static std::string catch_runons(const std::string& filename__);
static void find_manifast(const std::string& folderpath__);
// static std::map<std::string,std::string> find_manifast(const std::string& folderpath__);

// CpuInfo::CpuInfo(){
//     // devmclient_ = Singleton<DevmClient>::GetInstance();
// }

//using one interface to get all cpu info
CpuData CpuInfo::GetAllInfo() {
    CpuInfolscpu info_;
    CpuTemp temp_;
    std::map<std::string, std::string> binding_;
    std::vector<float> usage_;

    info_ = GetCpuInfo();
    temp_ = GetCpuTemp();
    binding_ = GetCpuBinding();
    usage_ = GetCpuUsage();

    cpu_data_.architecture = info_.architecture;
    cpu_data_.cpus = info_.cpus;
    cpu_data_.online_cpus = info_.online_cpus;
    cpu_data_.offline_cpus = info_.offline_cpus;
    cpu_data_.model_name = info_.model_name;
    cpu_data_.cpu_max_mhz = info_.cpu_max_mhz;
    cpu_data_.cpu_min_mhz = info_.cpu_min_mhz;
    cpu_data_.l1d_catch = info_.l1d_catch;
    cpu_data_.l1i_catch = info_.l1i_catch;
    cpu_data_.l2_catch = info_.l2_catch;
    cpu_data_.l3_catch = info_.l3_catch;
    cpu_data_.temp_cpu = temp_.temp_cpu;
    cpu_data_.temp_soc0 = temp_.temp_soc0;
    cpu_data_.temp_soc1 = temp_.temp_soc1;
    cpu_data_.temp_soc2 = temp_.temp_soc2;
    cpu_data_.cpu_binding = binding_;
    cpu_data_.cpus_usage = usage_;
    return cpu_data_;
}

//get mpstat
static std::string get_mpstat_output() {
    std::string output__;
    char buffer[S_SIZE];

#if defined(BUILD_FOR_ORIN)
    FILE* pipe = popen("/app/bin/mpstat -P ALL | awk '{print $3}' |sed -n '5,$p'", "r");
#else
    FILE* pipe = popen("mpstat -P ALL | awk '{print $3}' |sed -n '5,$p'", "r");
#endif
    if (!pipe) {
        DEVM_LOG_ERROR << "Error executing lscpu command.";
        return output__;
    }

    while (!feof(pipe)) {
        if (fgets(buffer, S_SIZE, pipe) != NULL) {
            output__ += buffer;
        }
    }

    pclose(pipe);
    return output__;
}

//获取各个CPU内核的使用率
std::vector<float> CpuInfo::GetCpuUsage() {

    std::string line;

    std::string output_ = get_mpstat_output();
    std::stringstream ss(output_);

    while (std::getline(ss, line)) {
        cpu_usage_.push_back(std::stof(line));
    }


    return cpu_usage_;
}

//获取CPU各个区域的温度值
CpuTemp CpuInfo::GetCpuTemp() {
    std::string line{};
    float temp{};
    try {
        //加载temp0
        temp = 0.0;
        line.clear();
        std::ifstream file(temppath0);
        if (file.is_open()) {
            std::getline(file, line);
            temp = std::stof(line);
            cpu_temp_.temp_cpu = temp / 1000;
            file.close();
        }

        //加载temp5
        temp = 0.0;
        line.clear();
        file.open(temppath5);
        if (file.is_open()) {
            std::getline(file, line);
            temp = std::stof(line);
            cpu_temp_.temp_soc0 = temp / 1000;
            file.close();
        }

        //加载temp6
        temp = 0.0;
        line.clear();
        file.open(temppath6);
        if (file.is_open()) {
            std::getline(file, line);
            temp = std::stof(line);
            cpu_temp_.temp_soc1 = temp / 1000;
            file.close();
        }

        //加载temp7
        temp = 0.0;
        line.clear();
        file.open(temppath7);
        if (file.is_open()) {
            std::getline(file, line);
            temp = std::stof(line);
            cpu_temp_.temp_soc2 = temp / 1000;
            file.close();
        }

    } catch (const std::exception& e) { DEVM_LOG_ERROR << "GetCpuTemp failed. failedCode: " << e.what(); }

    return cpu_temp_;
}

//把地址转换成服务名字
static std::string change_path_name(const std::string& path__) {
    std::string server_name_;

    size_t three_pos = path__.find('/', path__.find('/', path__.find('/') + 1) + 1);
    size_t four_pos = path__.find('/', three_pos + 1);
    if (three_pos != std::string::npos && four_pos != std::string::npos) {
        server_name_ = path__.substr(three_pos + 1, four_pos - three_pos - 1);
    }

    return server_name_;
}

//获取每个MANIFEST的run ons,用json解析
static std::string catch_runons(const std::string& filename__) {
    //打开json文件，获取内容
    std::ifstream file(filename__);
    json j;
    file >> j;
    file.close();

    std::string ons;
    std::stringstream ss;

    std::vector<int> runons = j["shall_run_ons"];
    //把数组转换成字符串输出
    for (long unsigned int i = 0; i < runons.size(); i++) {
        ss << runons[i];
        if (i != runons.size() - 1) {
            ss << " ";
        }
    }

    ons = ss.str();

    return ons;
}

//获取各个MANIFEST下面的run ons的值
// std::map<std::string,std::string> find_manifast(const std::string& folderpath__){
static void find_manifast(const std::string& folderpath__) {
    // std::map<std::string,std::string> strMap;
    std::string server_name;

    for (const auto& entry : fs::directory_iterator(folderpath__)) {
        if (entry.is_regular_file() && entry.path().filename() == "MANIFEST.json") {
            // strMap[folderpath__.c_str()] = catch_runons(entry.path());
            server_name = change_path_name(folderpath__);
            binding_map[server_name] = catch_runons(entry.path());

        } else if (entry.is_directory()) {
            find_manifast(entry.path());
        }
    }

    return;
}

std::map<std::string, std::string> CpuInfo::GetCpuBinding() {
    //获取run ons
    try {
        find_manifast(folderpath);
        cpu_binding_ = binding_map;
    } catch (const std::exception& e) { DEVM_LOG_ERROR << "can not find the folder" << e.what(); }
    return cpu_binding_;
}

//获取lscpu输出
static std::string get_lscpu_output() {
    std::string output__{};
    char buffer[MAX_SIZE];

    FILE* pipe = popen("lscpu", "r");
    if (!pipe) {
        DEVM_LOG_ERROR << "Error executing lscpu command.";
        return output__;
    }

    while (!feof(pipe)) {
        if (fgets(buffer, MAX_SIZE, pipe) != NULL) {
            output__ += buffer;
        }
    }

    pclose(pipe);
    return output__;
}

//获取对值
static std::string get_field_value(const std::string& output__, const std::string& token__) {
    std::istringstream iss(output__);
    std::string line;
    std::string value;

    while (std::getline(iss, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string val = line.substr(pos + 1);
            if (key.find(token__) != std::string::npos) {
                value = val;
                break;
            }
        }
    }

    return value;
}

//去除前缀空格
static std::string remove_prefix_space(const std::string& str) {
    size_t pos = str.find_first_not_of(' ');
    if (pos != std::string::npos) {
        return str.substr(pos);
    }
    return str;
}

CpuInfolscpu CpuInfo::GetCpuInfo() {
    std::string output_ = get_lscpu_output();
    std::string token;
    std::string value;
    DEVM_LOG_INFO << "after get_lscpu_output!";

    try {
        //On-line CPU(s) list
        token = "On-line CPU(s) list";
        value = get_field_value(output_, token);
        value = remove_prefix_space(value);
        cpu_info_.online_cpus = value;

        //Off-line CPU(s) list
        token = "Off-line CPU(s) list";
        value = get_field_value(output_, token);
        value = remove_prefix_space(value);
        cpu_info_.offline_cpus = value;
        DEVM_LOG_INFO << "after On Off!";
        if (getinfo_flag_ == 0) {
            //Architecture
            token = "Architecture";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.architecture = value;

            //CPU(s)
            token = "CPU(s)";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.cpus = std::stoi(value);

            //Model name
            token = "Model name";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.model_name = value;

            //CPU max MHz
            token = "CPU max MHz";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.cpu_max_mhz = std::stof(value);

            //CPU min MHz
            token = "CPU min MHz";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.cpu_min_mhz = std::stof(value);

            //L1d cache Kib
            token = "L1d cache";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.l1d_catch = std::stoi(value);

            //L1i cache Kib
            token = "L1i cache";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.l1i_catch = std::stoi(value);

            //L2 cache Mib
            token = "L2 cache";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.l2_catch = std::stoi(value);

            //L3 cache Mib
            token = "L3 cache";
            value = get_field_value(output_, token);
            value = remove_prefix_space(value);
            cpu_info_.l3_catch = std::stoi(value);

            getinfo_flag_ = 1;
            DEVM_LOG_INFO << "after getinfo_flag_ = 1!";
        }
    } catch (const std::exception& e) {
        DEVM_LOG_ERROR << "GetCpuInfo failed. failedCode: " << e.what();
    }

    return cpu_info_;
}

void CpuInfo::PrintCpuInfo() {
    if (getinfo_flag_ == 1) {
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "-------------------- CPUINFO --------------------" << std::endl;
        std::cout << std::setw(24) << std::left << "Architecture:" << cpu_info_.architecture << std::endl;
        std::cout << std::setw(24) << std::left << "CPU(s):" << cpu_info_.cpus << std::endl;
        std::cout << std::setw(24) << std::left << "On-line CPU(s) list:" << cpu_info_.online_cpus << std::endl;
        std::cout << std::setw(24) << std::left << "Off-line CPU(s) list:" << cpu_info_.offline_cpus << std::endl;
        std::cout << std::setw(24) << std::left << "Model name:" << cpu_info_.model_name << std::endl;
        std::cout << std::setw(24) << std::left << "CPU max MHz:" << cpu_info_.cpu_max_mhz << std::endl;
        std::cout << std::setw(24) << std::left << "CPU min MHz:" << cpu_info_.cpu_min_mhz << std::endl;
        std::cout << std::setw(24) << std::left << "L1d cache:" << cpu_info_.l1d_catch << " KiB" << std::endl;
        std::cout << std::setw(24) << std::left << "L1i cache:" << cpu_info_.l1i_catch << " KiB" << std::endl;
        std::cout << std::setw(24) << std::left << "L2 cache:" << cpu_info_.l2_catch << " MiB" << std::endl;
        std::cout << std::setw(24) << std::left << "L3 cache:" << cpu_info_.l3_catch << " MiB" << std::endl;
        std::cout << std::setw(24) << std::left << "Temperature:" << cpu_temp_.temp_cpu << "°C " << cpu_temp_.temp_soc0 << "°C " << cpu_temp_.temp_soc1 << "°C " << cpu_temp_.temp_soc2 << "°C " << std::endl;
        std::cout << std::setw(24) << std::left << "Cpu usage:" << "[";
        for (auto vec : cpu_usage_) {
            std::cout << vec << "% ";
        }
        std::cout << "]" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "--------------- proc binding core ---------------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        for (const auto& pair : cpu_binding_) {
            std::cout << std::setw(24) << std::left << pair.first << ": " << pair.second << std::endl;
        }
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
    }
}

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon