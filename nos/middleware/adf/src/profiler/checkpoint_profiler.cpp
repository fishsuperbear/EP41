#include "adf/include/profiler/checkpoint_profiler.h"

namespace hozon {
namespace netaos {
namespace adf {
CheckpointProfiler::CheckpointProfiler(const std::string& instance_name, uint32_t sample_size)
    : _instance_name(instance_name), _sample_size(sample_size), _period_us(_sample_size), _duration_us(_sample_size) {}

CheckpointProfiler::~CheckpointProfiler() {}

void CheckpointProfiler::Begin() {
    if (_checkpoints.empty()) {
        _checkpoints.emplace_back("BEGIN");
    }

    _checkpoints.front().Record();
}

void CheckpointProfiler::End() {
    if (_checkpoints.back().name != "END") {
        _checkpoints.emplace_back("END");
        _checkpoint_intervals_us.emplace_back(_sample_size);
        _checkpoints.back().Record();

        std::vector<std::string> names;
        for (auto cpt : _checkpoints) {
            names.emplace_back(cpt.name);
        }
        names.emplace_back("PERIOD");
        names.emplace_back("DURATION");

        std::cout << "names.size: " << names.size() << "\n";
        if (!_client.Init(_instance_name, PROFILER_SERVER_MULTICAST_ADDR, CHECKPOINT_MULTICAST_PORT, names)) {
            std::cout << "Fail to init profiler\n";
        }
    } else {
        _checkpoints.back().Record();
        CalcTimeDiffUs();
        std::vector<double> intervals;
        intervals.emplace_back(0);
        for (auto check_point_interval : _checkpoint_intervals_us) {
            intervals.emplace_back(check_point_interval.Back());
        }
        intervals.emplace_back(_period_us.Back());
        intervals.emplace_back(_duration_us.Back());
        std::cout << "interval.size: " << intervals.size() << "\n";
        _client.Send(intervals);
    }
}

void CheckpointProfiler::SetCheckPoint(const std::string& name) {
    if (_checkpoint_name_index_map.find(name) == _checkpoint_name_index_map.end()) {
        _checkpoint_name_index_map[name] = _checkpoints.size();
        _checkpoints.emplace_back(name);
        _checkpoint_intervals_us.emplace_back(_sample_size);
    }

    _checkpoints[_checkpoint_name_index_map[name]].Record();
}

void CheckpointProfiler::CalcTimeDiffUs() {
    for (uint32_t i = 0; i < _checkpoint_intervals_us.size(); ++i) {
        _checkpoint_intervals_us[i].Push(
            std::chrono::duration<double, std::milli>(_checkpoints[i + 1].time_point - _checkpoints[i].time_point)
                .count());
    }

    if (_period_us.Size() == 0) {
        _period_us.Push(0);
    } else {
        _period_us.Push(
            std::chrono::duration<double, std::milli>(_checkpoints[0].time_point - _checkpoints[0].old_time_point)
                .count());
    }

    _duration_us.Push(
        std::chrono::duration<double, std::milli>(_checkpoints.back().time_point - _checkpoints.front().time_point)
            .count());
}
}  // namespace adf
}  // namespace netaos
}  // namespace hozon