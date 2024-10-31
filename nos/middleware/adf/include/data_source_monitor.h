#pragma once

#include <time.h>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hozon {
namespace netaos {
namespace adf {

struct FreqInfo {
    uint64_t samples;
    double freq;
    uint64_t max_delta_us;
    uint64_t min_delta_us;
    double std_dev_us;
    uint64_t duration_us;
};

class BaseMonitor {
   public:
    virtual void ReadInfo() = 0;
};

class FreqMonitor : public BaseMonitor {
   public:
    FreqMonitor(const std::string& name, double exp_freq, double threshold_ratio, uint64_t max_samples = 1100);
    ~FreqMonitor();

    void EnablePeriodCheck();
    void DisablePeriodCheck();
    void Start();
    void Stop();
    void PushOnce();
    void ReadInfo();

   private:
    uint64_t GetCurrTimeStampUs();
    void PeriodCheck(uint64_t delta);

    std::string _name;
    std::mutex _mtx;
    bool _period_check;
    uint64_t _max_samples;
    std::vector<uint64_t> _ts_container;
    bool _pause;
    // double _exp_freq;
    double _period_us;
    double _threshold_ratio;
    uint64_t _last_read_time_us;
    uint64_t _total_cnt = 0;
    uint64_t _exeception_cnt = 0;
    // uint64_t _seq;
};

class MonitorReader {
   public:
    ~MonitorReader();
    static MonitorReader& GetInstance();
    void RegisterMonitor(BaseMonitor* monitor);
    void Stop();
    void SetFreqEnable(bool en);
    void SetReadPeriodMs(uint64_t period);

   private:
    MonitorReader();
    void Routine();

    std::mutex _mtx;
    std::condition_variable _cv;
    bool _running;
    uint64_t _read_period_ms;
    std::thread _th;
    std::vector<BaseMonitor*> _monitor_list;
    bool _freq_enable;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
