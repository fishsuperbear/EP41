/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: CostSpan definition
 */

#pragma once

#include <chrono>
#include "data_tools_logger.hpp"

namespace hozon {
namespace netaos {
namespace data_tool_common {

class CostCalc {
   public:
    CostCalc(const char* func, const char* op, uint64_t check_limit) : func_(func), op_(op), check_limit_(check_limit), start_(std::chrono::steady_clock::now()) {}

    ~CostCalc() {
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        uint64_t cost_us = std::chrono::microseconds(std::chrono::duration_cast<std::chrono::microseconds>(end - start_)).count();

        if (check_limit_ && cost_us < check_limit_) {
            return;
        }

        if (op_) {
            COMMON_LOG_WARN << "Time cost in operation " << op_ << " is " << (cost_us) << " us";
        } else {
            COMMON_LOG_WARN << "Time ost in function " << func_ << " is " << (cost_us) << " us";
        }
    }

   private:
    const char* func_;
    const char* op_;
    uint64_t check_limit_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

}
}
}

// #define COST_SPAN() CostSpan cs(__FUNCTION__, nullptr)
#define _LINE_STRINGIFY_(x) #x
#define LINE_STRINGIFY(x) _LINE_STRINGIFY_(x)
#define COST_CALC_FUNC() hozon::netaos::data_tool_common::CostCalc csf(__FUNCTION__, nullptr, 0)
#define COST_CALC_OP(op) hozon::netaos::data_tool_common::CostCalc cso(__FUNCTION__, op, 0)
#define COST_CALC_FUNC_WITH_CHECK(check_limit) hozon::netaos::data_tool_common::CostCalc csf(__FUNCTION__, nullptr, check_limit)
#define COST_CALC_OP_WITH_CHECK(op, check_limit) hozon::netaos::data_tool_common::CostCalc cso__LINE__(__FUNCTION__, op, check_limit)