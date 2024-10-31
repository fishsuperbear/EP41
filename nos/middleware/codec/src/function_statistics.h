/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: codec server loger
 */
#ifndef FUNCTION_STATISTICS_H_
#define FUNCTION_STATISTICS_H_
#pragma once

#include <chrono>
#include <iostream>

#include "codec/src/codec_logger.h"
namespace hozon {
namespace netaos {
namespace codec {
class FunctionStatistics {
   public:
    FunctionStatistics(std::string function_name) {
        startTime = std::chrono::steady_clock::now();
        function_name_ = function_name;
    }

    ~FunctionStatistics() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        CODEC_DEBUG << function_name_ << " execution time: " << duration.count() << " ms";
    }

   private:
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::string function_name_;
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif
