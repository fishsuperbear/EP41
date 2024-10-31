#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include "adf-lite/include/base.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class Bundle {
public:
    Bundle();
    ~Bundle();

    std::vector<BaseDataTypePtr> GetAll(const std::string& topic);
    BaseDataTypePtr GetOne(const std::string& topic);
    BaseDataTypePtr GetOne(const std::string& topic, uint64_t optimal_time_us, uint64_t min_time_us, uint64_t max_time_us);
    void Add(const std::string& topic, BaseDataTypePtr data);
    void Set(const std::string& topic, std::vector<BaseDataTypePtr>& data_vec);
    std::unordered_map<std::string, std::vector<BaseDataTypePtr>>& GetRaw();
    std::vector<std::pair<std::string, BaseDataTypePtr>> GetInTimeOrder();

private:
    std::unordered_map<std::string, std::vector<BaseDataTypePtr>> _data_map;
};

}
}
}

