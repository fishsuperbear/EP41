#include "adf/include/ts_align/ts_align_impl.h"
#include <algorithm>
#include <cstdint>
#include <iostream>

namespace hozon {
namespace netaos {
namespace adf {

int32_t TsAlignImpl::Init(uint32_t time_window_ms, uint32_t validity_time_ms, TsAlign::AlignSuccFunc func) {
    _time_window_ms = time_window_ms;
    _validity_time_ms = validity_time_ms;
    _align_succ_func = func;

    _need_stop = false;
    _align_thread = std::make_shared<std::thread>(&TsAlignImpl::Routine, this);

    return 0;
}

void TsAlignImpl::Deinit() {
    _need_stop = true;
    _align_data_arrived_cv.notify_all();
    _align_thread->join();
}

void TsAlignImpl::RegisterSource(const std::string& source_name) {
    _align_mutex.lock();
    _sources.emplace_back(source_name);
    _source_latest_timestamp[source_name] = 0;
    _align_mutex.unlock();
}

void TsAlignImpl::Push(const std::string& source_name, TsAlignDataType data, uint64_t timestamp_us) {
    std::pair<std::string, std::shared_ptr<DataWithTs>> data_pair;
    data_pair.first = source_name;
    data_pair.second = std::make_shared<DataWithTs>();
    std::shared_ptr<DataWithTs>& dt = data_pair.second;
    dt->data = data;
    dt->timestamp_us = timestamp_us;

    std::unique_lock<std::mutex> lk(_align_mutex);
    if (dt->timestamp_us > _source_latest_timestamp[source_name]) {
        _source_latest_timestamp[source_name] = dt->timestamp_us;
    }

    auto lower_it = std::lower_bound(_align_list.begin(), _align_list.end(), data_pair,
                                     [](std::pair<std::string, std::shared_ptr<DataWithTs>> A,
                                        std::pair<std::string, std::shared_ptr<DataWithTs>> B) {
                                         return A.second->timestamp_us < B.second->timestamp_us;
                                     });
    _align_list.insert(lower_it, data_pair);

    for (auto it = _align_list.begin(); it != _align_list.end();) {
        if (it->first != source_name) {
            ++it;
            continue;
        }

        uint64_t vtime = 0;
        if (_source_latest_timestamp[source_name] > _validity_time_ms * 1000) {
            vtime = _source_latest_timestamp[source_name] - _validity_time_ms * 1000;
        }

        if (it->second->timestamp_us < vtime) {
            it = _align_list.erase(it);
        } else {
            ++it;
        }
    }
    _align_data_arrived_cv.notify_all();
}

int32_t TsAlignImpl::CheckDataAligned(std::unordered_map<std::string, std::shared_ptr<DataWithTs>>& data_map) {
    uint64_t max_us = 0;
    uint64_t min_us = UINT64_MAX;

    for (auto& source : _sources) {
        if (data_map.find(source) == data_map.end()) {
            // std::cout << "Aligning timestamp, missing data " << source << std::endl;
            return -1;
        }
    }

    for (auto& source : _sources) {
        max_us = std::max(data_map[source]->timestamp_us, max_us);
        min_us = std::min(data_map[source]->timestamp_us, min_us);
        // std::cout << "Aligning timestamp of " << source <<
        //                 ", curr: " << data_map[source]->timestamp_us / 1000
        //                 << " min: " << min_us / 1000 << " max: " << max_us / 1000 << std::endl;
    }

    if ((max_us - min_us) > (_time_window_ms * 1000)) {
        // std::cout << "Aligning timestamp failed, time diff " << (max_us - min_us) / 1000 << std::endl;
        return -1;
    }

    // std::cout << "Aligning timestamp succ, time diff " << (max_us - min_us) / 1000 << std::endl;

    return 0;
}

int32_t TsAlignImpl::AlignSources(TsAlignDataBundle& bundle) {
    std::unordered_map<std::string, std::shared_ptr<DataWithTs>> data_map;

    for (auto rit = _align_list.rbegin(); rit != _align_list.rend(); ++rit) {
        data_map[rit->first] = rit->second;
        int ret = CheckDataAligned(data_map);
        if (ret == 0) {
            for (auto& ele : data_map) {
                bundle[ele.first] = ele.second->data;

                // TODO: use map::find
                for (auto it = _align_list.begin(); it != _align_list.end();) {
                    if (ele.second == it->second) {
                        it = _align_list.erase(it);
                        break;
                    } else {
                        ++it;
                    }
                }
            }
            return 0;
        }
    }

    return -1;
}

int32_t TsAlignImpl::GetAligned(TsAlignDataBundle& bundle) {
    std::unique_lock<std::mutex> align_lk(_align_mutex);

    // ADF_LOG_TRACE << "Try to get aligned sources of " << trigger.name;
    int32_t ret = AlignSources(bundle);
    if (ret == 0) {
        // ADF_LOG_TRACE << "Get aligned sources from existing data " << trigger.name;
        return 0;
    }
    // std::cout << "first align failed\n";
    while (!_need_stop) {
        _align_data_arrived_cv.wait(align_lk);

        ret = AlignSources(bundle);
        if (ret == 0) {
            return 0;
        }
    }

    return -1;
}

void TsAlignImpl::Routine() {
    while (!_need_stop) {
        TsAlignDataBundle bundle;
        int32_t ret = GetAligned(bundle);
        if (ret < 0) {
            continue;
        }

        if (_align_succ_func) {
            _align_succ_func(bundle);
        }
    }
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon