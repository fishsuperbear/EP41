/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: CostSpan definition
 */

#ifndef V2C_DATA_COLLECT_COMMON_COST_SPAN_H
#define V2C_DATA_COLLECT_COMMON_COST_SPAN_H

#include <chrono>
#include "camera_venc_logger.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

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
            CAMV_DEBUG << "Time cost in operation " << op_ << " is " << (cost) << " us";
        } else {
            CAMV_DEBUG << "Time ost in function " << func_ << " is " << (cost) << " us";
        }
    }

   private:
    const char* func_;
    const char* op_;
    uint64_t check_limit_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

// #define COST_SPAN() CostSpan cs(__FUNCTION__, nullptr)
#define COST_CALC_FUNC() CostSpan csf(__FUNCTION__, nullptr, 0)
#define COST_CALC_OP(op) CostSpan cso(__FUNCTION__, op, 0)
#define COST_CALC_FUNC_WITH_CHECK(check_limit) CostSpan csf(__FUNCTION__, nullptr, check_limit)
#define COST_CALC_OP_WITH_CHECK(op, check_limit) CostSpan cso(__FUNCTION__, op, check_limit)

}
}
}

#endif