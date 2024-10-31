#include "adf/include/data_source_monitor.h"
#include "adf/include/internal_log.h"

namespace hozon {
namespace netaos {
namespace adf {
template <typename T>
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

FreqMonitor::FreqMonitor(const std::string& name, double exp_freq, double threshold_ratio, uint64_t max_samples)
    : _name(name),
      _period_check(false),
      _max_samples(max_samples),
      _pause(true),
      _period_us(1000 * 1000 / exp_freq),
      _threshold_ratio(threshold_ratio) {
    _ts_container.reserve(_max_samples);
}

FreqMonitor::~FreqMonitor() {
    if (_period_check) {
        ADF_LOG_INFO << "Freq monitor " << _name << " exited, exception counter " << _exeception_cnt << "/"
                     << _total_cnt;
    }
}

void FreqMonitor::EnablePeriodCheck() {
    _period_check = true;
}

void FreqMonitor::DisablePeriodCheck() {
    _period_check = false;
}

void FreqMonitor::Start() {
    std::lock_guard<std::mutex> lk(_mtx);
    _last_read_time_us = GetCurrTimeStampUs();
    _pause = false;
    _ts_container.clear();
}

void FreqMonitor::Stop() {
    std::lock_guard<std::mutex> lk(_mtx);
    _pause = true;
}

void FreqMonitor::PushOnce() {
    std::lock_guard<std::mutex> lk(_mtx);
    uint64_t curr_time_us = GetCurrTimeStampUs();
    if (_period_check && !_ts_container.empty()) {
        PeriodCheck(curr_time_us - _ts_container.back());
    }

    _ts_container.emplace_back(curr_time_us);
}

void FreqMonitor::ReadInfo() {
    FreqInfo info;
    uint64_t curr_time_us = GetCurrTimeStampUs();
    uint64_t max_delta_us = 0;
    uint64_t min_delta_us = UINT64_MAX;
    std::vector<uint64_t> delta_us_vec;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (_pause) {
            return;
        }

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

    ADF_LOG_INFO << "Freq info: " << _name << ", " << info.freq << ", " << info.max_delta_us / 1000 << ", "
                 << info.min_delta_us / 1000 << ", " << info.std_dev_us / 1000 << ", " << info.duration_us / 1000
                 << ", " << info.samples << ", " << _exeception_cnt << "/" << _total_cnt;
}

uint64_t FreqMonitor::GetCurrTimeStampUs() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);

    return time.tv_sec * 1000 * 1000 + time.tv_nsec / 1000;
}

void FreqMonitor::PeriodCheck(uint64_t delta) {
    if (!_period_check) {
        return;
    }
    double upper = _period_us * (1 + _threshold_ratio);
    double lower = _period_us * (1 - _threshold_ratio);
    ++_total_cnt;

    if ((delta > upper) || (delta < lower)) {
        ++_exeception_cnt;
        // ADF_LOG_WARN << "Period exception occurs, name: " << _name
        //     << ", expected period: " << (double)(_period_us) / 1000
        //     << "(ms), delta " << (double)(delta) / 1000
        //     << "(ms), period_exception " << _exeception_cnt
        //     << "/" << _total_cnt;
    }
}

MonitorReader::~MonitorReader() {}

MonitorReader& MonitorReader::GetInstance() {
    static MonitorReader reader;
    return reader;
}

void MonitorReader::RegisterMonitor(BaseMonitor* monitor) {
    std::lock_guard<std::mutex> lk(_mtx);
    _monitor_list.emplace_back(monitor);
}

void MonitorReader::Stop() {
    _running = false;
    _cv.notify_all();
    _th.join();
}

void MonitorReader::SetFreqEnable(bool en) {
    _freq_enable = en;
}

void MonitorReader::SetReadPeriodMs(uint64_t period) {
    _read_period_ms = period;
}

MonitorReader::MonitorReader()
    : _running(true), _read_period_ms(5000), _th(&MonitorReader::Routine, this), _freq_enable(true) {}

void MonitorReader::Routine() {
    while (_running) {
        std::unique_lock<std::mutex> lk(_mtx);
        _cv.wait_for(lk, std::chrono::milliseconds(_read_period_ms));

        if (!_freq_enable) {
            continue;
        }

        ADF_LOG_INFO << "Freq monitor info: "
                     << "name"
                     << ", freq"
                     << ", d_max"
                     << "(ms), d_min"
                     << "(ms), std_dev"
                     << "(ms), duration"
                     << "(ms), samples"
                     << ", period_exception";

        for (auto& monitor : _monitor_list) {
            monitor->ReadInfo();
        }
    }
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
