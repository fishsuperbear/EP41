#ifndef FUNCTION_STATISTICS_H_
#define FUNCTION_STATISTICS_H_
#pragma once

#include <chrono>
#include <iostream>

#include "system_monitor/include/common/system_monitor_logger.h"
namespace hozon {
namespace netaos {
namespace system_monitor {
class FunctionStatistics {
 public:
    FunctionStatistics(std::string function_name) {
        startTime = std::chrono::steady_clock::now();
        function_name_ = function_name;
    }

    ~FunctionStatistics() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        STMM_INFO << function_name_ << " execution time: " << duration.count() << " us";
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::string function_name_;
};
}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif
