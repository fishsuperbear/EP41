#pragma once

#include <chrono>
#include <iostream>

#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {
class FunctionStatistics {
 public:
    FunctionStatistics(std::string function_name) {
        startTime = std::chrono::steady_clock::now();
        function_name_ = function_name;
    }

    ~FunctionStatistics() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        UM_INFO << function_name_ << " execution time: " << duration.count() << " us";
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::string function_name_;
};
}  // namespace update
}  // namespace netaos
}  // namespace hozon
