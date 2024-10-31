#include "adf-lite/include/executor.h"
#include "config_param.h"
#include "adf/include/log.h"
#include "adf-lite/include/data_types/image/orin_image.h"

using namespace hozon::netaos::adf_lite;
using namespace hozon::netaos::cfg;
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

class NvsSensorExecutor : public Executor {
   public:
    NvsSensorExecutor();
    ~NvsSensorExecutor();

    int32_t AlgInit();
    void AlgRelease();

   private:
    int32_t NvsCamProcess(Bundle* input);
    int32_t EncodeImageProcess(Bundle* input);
    int32_t ProtoImageProcess(Bundle* input);

    template <typename T>
        std::shared_ptr<T> GetImageData(Bundle* input, const std::string& name);
    void WriteFile(const std::string& name, uint8_t* data, uint32_t size);
    void DumpCudaImage(const std::string& name, std::shared_ptr<NvsImageCUDA> packet);

    FreqChecker freq_checker;
};

REGISTER_ADF_CLASS(cam_adf_lite, NvsSensorExecutor)