
#include "adf/include/node_bundle.h"
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>
#include "adf/include/data_types/common/types.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeBundle::NodeBundle() {}

NodeBundle::~NodeBundle() {}

std::vector<BaseDataTypePtr> NodeBundle::GetAll(const std::string& name) {
    const auto& it = _data_map.find(name);
    if (it == _data_map.end()) {
        return std::vector<BaseDataTypePtr>();
    }

    return it->second;
}

BaseDataTypePtr NodeBundle::GetOne(const std::string& name) {
    const auto& it = _data_map.find(name);
    if (it == _data_map.end()) {
        return nullptr;
    }

    if (it->second.empty()) {
        return nullptr;
    }

    return it->second[0];
}

uint64_t CalcTimeDiffAbs(uint64_t t1, uint64_t t2) {
    if (t1 >= t2) {
        return t1 - t2;
    } else {
        return t2 - t1;
    }
}

BaseDataTypePtr NodeBundle::GetOne(const std::string& name, uint64_t optimal_time_us, uint64_t min_time_us,
                                   uint64_t max_time_us) {
    const auto& it = _data_map.find(name);
    if (it == _data_map.end()) {
        return nullptr;
    }

    BaseDataTypePtr optimal_one = nullptr;
    uint64_t optimal_time_diff = 0;
    for (auto sample : it->second) {
        if ((sample->__header.timestamp_real_us >= min_time_us) &&
            (sample->__header.timestamp_real_us <= max_time_us)) {
            if (optimal_one == nullptr) {
                optimal_one = sample;
                optimal_time_diff = CalcTimeDiffAbs(sample->__header.timestamp_real_us, optimal_time_us);
            } else {
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

void NodeBundle::Add(const std::string& name, BaseDataTypePtr data) {
    _data_map[name].emplace_back(data);
}

void NodeBundle::Set(const std::string& name, std::vector<BaseDataTypePtr>& data_vec) {
    _data_map[name] = data_vec;
}

std::unordered_map<std::string, std::vector<BaseDataTypePtr>>& NodeBundle::GetRaw() {
    return _data_map;
}

std::vector<std::pair<std::string, BaseDataTypePtr>> NodeBundle::GetInTimeOrder() {
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

    std::sort(time_ordered_vec.begin(), time_ordered_vec.end(),
              [](typeof(*time_ordered_vec.begin()) A, typeof(*time_ordered_vec.end()) B) {
                  return A.second->__header.timestamp_real_us < B.second->__header.timestamp_real_us;
              });

    return time_ordered_vec;
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon