#ifndef _CPUINFO_INTERFACE_H_
#define _CPUINFO_INTERFACE_H_
#include <map>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace devm_server {

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

struct CpuInfolscpu {
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
};

struct CpuTemp {
    float temp_cpu;
    float temp_soc0;
    float temp_soc1;
    float temp_soc2;
};

class CpuInfo {
   public:
    CpuInfo() {}
    ~CpuInfo() {}
    CpuInfolscpu GetCpuInfo();
    CpuTemp GetCpuTemp();
    std::map<std::string, std::string> GetCpuBinding();
    std::vector<float> GetCpuUsage();
    CpuData GetAllInfo();
    void PrintCpuInfo();

   private:
    CpuInfolscpu cpu_info_{};
    CpuTemp cpu_temp_{};
    std::vector<float> cpu_usage_{};
    std::map<std::string, std::string> cpu_binding_{};
    CpuData cpu_data_{};
    volatile int16_t getinfo_flag_ = 0;
};

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
#endif