/* *
 * FUNCTION: Define Common Types
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 */

#pragma once

#include <string>
#include <unordered_map>


struct AlgTime {
    /* data */
    uint64_t sec;
    uint64_t nsec;
};

namespace hozon {
namespace netaos {
namespace adf {

struct LatencyInformation {
    std::unordered_map<std::string, struct AlgTime> data;

    LatencyInformation& operator=(const LatencyInformation& another) {
        data = another.data;
        return *this;
    }
};


}    
}
}