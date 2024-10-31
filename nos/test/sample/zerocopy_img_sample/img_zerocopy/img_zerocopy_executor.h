#include "adf-lite/include/data_types/image/orin_image.h"
#include "adf-lite/include/executor.h"
#include "adf/include/log.h"
#include "config_param.h"

using namespace hozon::netaos::adf_lite;

class FreqChecker {
    using checker_time = std::chrono::time_point<std::chrono::system_clock>;

   public:
    FreqChecker() = default;
    void say(const std::string& unique_name, uint64_t sample_cnt = 100);

   private:
    std::unordered_map<std::string, std::pair<uint64_t, checker_time>> freq_map_;
};

void FreqChecker::say(const std::string& unique_name, uint64_t sample_cnt) {
    if (freq_map_.find(unique_name) == freq_map_.end()) {
        freq_map_[unique_name] = std::make_pair(1, std::chrono::system_clock::now());
    } else {
        freq_map_[unique_name].first++;
    }

    if (freq_map_[unique_name].first == sample_cnt) {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = now - freq_map_[unique_name].second;
        NODE_LOG_INFO << "check " << unique_name << " frequency: " << sample_cnt / diff.count() << " Hz";
        freq_map_[unique_name].second = now;
        freq_map_[unique_name].first = 0;
    }
}

class IdlCudaExecutor : public Executor {
   public:
    IdlCudaExecutor();
    ~IdlCudaExecutor();

    int32_t AlgInit();
    void AlgRelease();

   private:
    int32_t NVSCamProcess(Bundle* input);
    int32_t DumpNVSCamProcess(Bundle* input);
    FreqChecker freq_checker;
    BaseDataTypePtr _lastet_data;

    void WriteFile(const std::string& name, uint8_t* data, uint32_t size);
    void ImageDumpFile(const std::string& file_name, int index, std::shared_ptr<NvsImageCUDA> packet);
    bool dump_file = false;
    uint32_t cam_idx = 0;
};

REGISTER_ADF_CLASS(idl_cuda_test, IdlCudaExecutor)