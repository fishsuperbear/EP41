#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace adf {

class SimpleFreqChecker {
    using checker_time = std::chrono::time_point<std::chrono::system_clock>;
    using Callback = std::function<void(const std::string&, double)>;

   public:
    SimpleFreqChecker(Callback callback) { _callback = callback; }

    void say(const std::string& unique_name, uint64_t sample_cnt = 100) {
        if (freq_map_.find(unique_name) == freq_map_.end()) {
            freq_map_[unique_name] = std::make_pair(1, std::chrono::system_clock::now());
        } else {
            freq_map_[unique_name].first++;
        }

        if (freq_map_[unique_name].first == sample_cnt) {
            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = now - freq_map_[unique_name].second;
            if (_callback) {
                _callback(unique_name, sample_cnt / diff.count());
                // _logger->LogInfo() << "Check " << unique_name << " frequency: " << sample_cnt / diff.count() << " Hz";
            }
            freq_map_[unique_name].second = now;
            freq_map_[unique_name].first = 0;
        }
    }

   private:
    std::unordered_map<std::string, std::pair<uint64_t, checker_time>> freq_map_;
    Callback _callback;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon