#pragma once

#include <string.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>
#include "adf/include/profiler/profiler_client.h"

namespace hozon {
namespace netaos {
namespace adf {
template <typename T>
class SizeLimitedContainer {
   public:
    SizeLimitedContainer(uint32_t size) { _size_limit = size; }

    virtual ~SizeLimitedContainer() {}

    void Push(const T& data) {
        if (_container.size() >= _size_limit) {
            _sum -= _container.front();
            _container.pop_front();
        }

        _container.emplace_back(data);
        _sum += data;
    }

    T Avg() { return _sum / _container.size(); }

    uint32_t Size() { return _container.size(); }

    T& Back() { return _container.back(); }

    T& operator[](uint32_t index) { return _container[index]; }

   protected:
    uint32_t _size_limit;
    std::deque<T> _container;
    T _sum;
};

struct CheckPoint {
    CheckPoint(const std::string& _name) : name(_name) {}

    void Record() {
        old_time_point = time_point;
        time_point = std::chrono::steady_clock::now();
    }

    std::string name;
    std::chrono::steady_clock::time_point time_point = std::chrono::steady_clock::time_point::min();
    std::chrono::steady_clock::time_point old_time_point = std::chrono::steady_clock::time_point::min();
};

class CheckpointProfiler {
   public:
    CheckpointProfiler(const std::string& instance_name, uint32_t sample_size = 1000);
    ~CheckpointProfiler();

    void Begin();
    void End();
    void SetCheckPoint(const std::string& name);

   protected:
    void CalcTimeDiffUs();

   private:
    std::string _instance_name;
    uint32_t _sample_size;
    std::vector<CheckPoint> _checkpoints;
    std::vector<SizeLimitedContainer<double>> _checkpoint_intervals_us;
    SizeLimitedContainer<double> _period_us;
    SizeLimitedContainer<double> _duration_us;
    std::unordered_map<std::string, uint32_t> _checkpoint_name_index_map;
    ProfilerClient<double> _client;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon