/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-11-14 09:11:46
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-11-14 09:41:40
 * @FilePath: /nos/middleware/cfg/src/function_statistics.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: codec server loger
 */
#ifndef FUNCTION_STATISTICS_H_
#define FUNCTION_STATISTICS_H_
#pragma once

#include <chrono>
#include <iostream>

#include "sys_statemgr/include/logger.h"

namespace hozon {
namespace netaos {
namespace ssm {
class FunctionStatistics {
 public:
    FunctionStatistics(std::string function_name) {
        startTime = std::chrono::steady_clock::now();
        function_name_ = function_name;
    }

    ~FunctionStatistics() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        SSM_LOG_ERROR << function_name_ << " execution time: " << duration.count() << " us";
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::string function_name_;
};
}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif
