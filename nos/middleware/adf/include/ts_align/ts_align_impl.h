#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "adf/include/ts_align/ts_align.h"

namespace hozon {
namespace netaos {
namespace adf {

class TsAlignImpl {
   public:
    int32_t Init(uint32_t time_window_ms, uint32_t validity_time_ms, TsAlign::AlignSuccFunc func);
    void Deinit();
    void RegisterSource(const std::string& source_name);
    void Push(const std::string& source_name, TsAlignDataType data, uint64_t timestamp_us);

   private:
    struct DataWithTs {
        TsAlignDataType data;
        uint64_t timestamp_us;
    };

    int32_t GetAligned(TsAlignDataBundle& bundle);
    int32_t AlignSources(TsAlignDataBundle& bundle);
    int32_t CheckDataAligned(std::unordered_map<std::string, std::shared_ptr<DataWithTs>>& data_map);
    void Routine();

    uint32_t _time_window_ms;
    uint32_t _validity_time_ms;
    TsAlign::AlignSuccFunc _align_succ_func;
    std::vector<std::string> _sources;
    bool _need_stop = false;
    std::shared_ptr<std::thread> _align_thread;

    std::mutex _align_mutex;
    std::condition_variable _align_data_arrived_cv;
    std::list<std::pair<std::string, std::shared_ptr<DataWithTs>>> _align_list;
    std::unordered_map<std::string, uint64_t> _source_latest_timestamp;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon