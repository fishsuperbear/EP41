#include "adf-lite/include/executor.h"
#include "config_param.h"
#include "adf/include/log.h"
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

class FisheyePerceptionExecutor : public Executor {
   public:
    FisheyePerceptionExecutor();
    ~FisheyePerceptionExecutor();

    int32_t AlgInit();
    void AlgRelease();

   private:
    int32_t ImageProcess(Bundle* input);
    int32_t Workflow1(Bundle* input);
    int32_t Workflow2(Bundle* input);
    int32_t Workflow3(Bundle* input);
    int32_t ReceiveCmTopic(Bundle* input);
    int32_t ReceiveLinkSample(Bundle* input, const ProfileToken& token);
    int32_t ReceiveStructTopic(Bundle* input);
    int32_t ReceivePlainStructTopic(Bundle* input);
    int32_t ReceiveNotPlainStructTopic(Bundle* input);
    int32_t ReceiveStatusChange(Bundle* input);
    int32_t FreeDataTopic(Bundle* input);
    int32_t FisheyeEventCheck1(Bundle* input);
    int32_t FisheyeEventCheck2(Bundle* input);
    int32_t GetLatestData(hozon::netaos::adf_lite::BaseDataTypePtr& ptr);
    int32_t NVSCamProcess(Bundle* input);
    FreqChecker freq_checker;
    BaseDataTypePtr _lastet_data;
    int32_t _recv_status = 0;
};

REGISTER_ADF_CLASS(FisheyePerception, FisheyePerceptionExecutor)