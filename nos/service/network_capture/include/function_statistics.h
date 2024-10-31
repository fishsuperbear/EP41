/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: network_capture FunctionStatistics
 */
#ifndef NETWORK_CAPTURE_FUNCTION_STATISTICS_H_
#define NETWORK_CAPTURE_FUNCTION_STATISTICS_H_
#pragma once

#include <chrono>
#include <iostream>

#include "network_capture/include/network_logger.h"
namespace hozon {
namespace netaos {
namespace network_capture {
class FunctionStatistics {
   public:
    FunctionStatistics(std::string function_name) {
        startTime = std::chrono::steady_clock::now();
        function_name_ = function_name;
    }

    ~FunctionStatistics() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        NETWORK_LOG_DEBUG << function_name_ << " execution time: " << duration.count() << " ns";
    }

   private:
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::string function_name_;
};
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
#endif
