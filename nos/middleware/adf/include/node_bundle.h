#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "adf/include/data_types/common/types.h"

namespace hozon {
namespace netaos {
namespace adf {
class NodeBundle {
   public:
    NodeBundle();
    ~NodeBundle();

    std::vector<BaseDataTypePtr> GetAll(const std::string& name);
    BaseDataTypePtr GetOne(const std::string& name);
    BaseDataTypePtr GetOne(const std::string& name, uint64_t optimal_time_us, uint64_t min_time_us,
                           uint64_t max_time_us);
    void Add(const std::string& name, BaseDataTypePtr data);
    void Set(const std::string& name, std::vector<BaseDataTypePtr>& data_vec);
    std::unordered_map<std::string, std::vector<BaseDataTypePtr>>& GetRaw();
    std::vector<std::pair<std::string, BaseDataTypePtr>> GetInTimeOrder();

   private:
    std::unordered_map<std::string, std::vector<BaseDataTypePtr>> _data_map;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon