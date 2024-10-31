#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <cmath>
#include <time.h>

namespace hozon {
namespace netaos {
namespace adf_lite {

struct FreqInfo {
    uint64_t samples;
    double freq;
    uint64_t max_delta_us;
    uint64_t min_delta_us;
    double std_dev_us;
    uint64_t duration_us;
};

class FreqSta {
public:
    FreqSta(uint64_t max_samples = 1100) :
            _max_samples(max_samples) {
        _ts_container.reserve(_max_samples);
    }
    ~FreqSta() {}

    void Start() {
        std::lock_guard<std::mutex> lk(_mtx);
        _last_read_time_us = GetCurrTimeStampUs();
        _ts_container.clear();
    }

    void PushOnce() {
        std::lock_guard<std::mutex> lk(_mtx);
        uint64_t curr_time_us = GetCurrTimeStampUs();
        _ts_container.emplace_back(curr_time_us);
    }

    FreqInfo ReadInfo() {
        FreqInfo info;
        uint64_t curr_time_us = GetCurrTimeStampUs();
        uint64_t max_delta_us = 0;
        uint64_t min_delta_us = UINT64_MAX;
        std::vector<uint64_t> delta_us_vec;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            
            delta_us_vec.reserve(_ts_container.size());
            for (std::size_t i = 1; i < _ts_container.size(); ++i) {
                uint64_t delta_us = _ts_container[i] - _ts_container[i - 1];
                delta_us_vec.emplace_back(delta_us);
                max_delta_us = std::max(max_delta_us, delta_us);
                min_delta_us = std::min(min_delta_us, delta_us);
            }

            info.samples = _ts_container.size();
            _ts_container.clear();
        }

        info.duration_us = curr_time_us - _last_read_time_us;
        info.freq = info.samples * 1000000.0f / info.duration_us;
        info.max_delta_us = max_delta_us;
        info.min_delta_us = (min_delta_us == UINT64_MAX) ? 0 : min_delta_us;
        info.std_dev_us = CalcStdDev(delta_us_vec);
        _last_read_time_us = curr_time_us;

        return info;
    }

private:
    uint64_t GetCurrTimeStampUs() {
        struct timespec time;
        clock_gettime(CLOCK_REALTIME, &time);

        return time.tv_sec * 1000 * 1000 + time.tv_nsec / 1000;
    }

    template<typename T>
    double CalcStdDev(std::vector<T>& data) {
        double sum = 0;

        for (auto& ele : data) {
            sum += ele;
        }

        double mean = sum / data.size();
        double std_dev = 0;
        for (auto& ele : data) {
            std_dev = std::pow(mean - ele, 2);
        }

        return std::sqrt(std_dev / data.size());
    }

    std::mutex _mtx;
    uint64_t _max_samples;
    std::vector<uint64_t> _ts_container;
    uint64_t _last_read_time_us;
};

}
}
}
