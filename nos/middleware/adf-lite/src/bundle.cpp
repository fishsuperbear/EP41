#include "adf-lite/include/bundle.h"
#include <algorithm>

namespace hozon {
namespace netaos {
namespace adf_lite {

Bundle::Bundle() {

}

Bundle::~Bundle() {

}

std::vector<BaseDataTypePtr> Bundle::GetAll(const std::string& topic) {
    if (_data_map.find(topic) == _data_map.end()) {
        return std::vector<BaseDataTypePtr>();
    }

    return _data_map[topic];
}

BaseDataTypePtr Bundle::GetOne(const std::string& topic) {
    if (_data_map.find(topic) == _data_map.end()) {
        return nullptr;
    }

    if (_data_map[topic].empty()) {
        return nullptr;
    }

    return _data_map[topic][0];
}

uint64_t CalcTimeDiffAbs(uint64_t t1, uint64_t t2) {
    if (t1 >= t2) {
        return t1 - t2;
    }
    else {
        return t2 - t1;
    }
}

BaseDataTypePtr Bundle::GetOne(const std::string& topic, uint64_t optimal_time_us, uint64_t min_time_us, uint64_t max_time_us) {
    if (_data_map.find(topic) == _data_map.end()) {
        return nullptr;
    }

    BaseDataTypePtr optimal_one = nullptr;
    uint64_t optimal_time_diff = 0;
    for (auto sample : _data_map[topic]) {
        if ((sample->__header.timestamp_real_us >= min_time_us) && (sample->__header.timestamp_real_us <= max_time_us)) {
            if (optimal_one == nullptr) {
                optimal_one = sample;
                optimal_time_diff = CalcTimeDiffAbs(sample->__header.timestamp_real_us, optimal_time_us);
            }
            else {
                uint64_t tmp_diff = CalcTimeDiffAbs(sample->__header.timestamp_real_us, optimal_time_us);
                if (tmp_diff < optimal_time_diff) {
                    optimal_one = sample;
                    optimal_time_diff = tmp_diff;
                }
            }
        }
    }

    return optimal_one;
}

void Bundle::Add(const std::string& topic, BaseDataTypePtr data) {
    _data_map[topic].emplace_back(data);
}

void Bundle::Set(const std::string& topic, std::vector<BaseDataTypePtr>& data_vec) {
    _data_map[topic] = data_vec;
}

std::unordered_map<std::string, std::vector<BaseDataTypePtr>>& Bundle::GetRaw() {
    return _data_map;
}

std::vector<std::pair<std::string, BaseDataTypePtr>> Bundle::GetInTimeOrder() {
    std::vector<std::pair<std::string, BaseDataTypePtr>> time_ordered_vec;
    std::size_t size = 0;
    
    for (auto& ele : _data_map) {
        size += ele.second.size();
    }
    time_ordered_vec.reserve(size);

    for (auto& ele : _data_map) {
        for (auto& data : ele.second) {
            time_ordered_vec.emplace_back(std::make_pair(ele.first, data));
        }
    }

    std::sort(time_ordered_vec.begin(), time_ordered_vec.end(), [](__typeof__(*time_ordered_vec.begin()) A, __typeof__(*time_ordered_vec.end()) B) {
        return A.second->__header.timestamp_real_us < B.second->__header.timestamp_real_us;
    });

    return time_ordered_vec;
}

}
}
}