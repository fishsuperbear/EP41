#pragma once

#include <chrono>
#include <iostream>

#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {
class FunctionStatistics {
 public:
    FunctionStatistics(std::string function_name) {
        startTime = std::chrono::steady_clock::now();
        function_name_ = function_name;
    }

    ~FunctionStatistics() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        LOG_SERVER_INFO << function_name_ << " execution time: " << duration.count() << " us";
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::string function_name_;
};
}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
