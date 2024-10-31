/* *
 * FUNCTION: Define Common Types
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 */

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "idl/generated/common.h"
#include "proto/common/header.pb.h"
#include "adf/include/data_types/common/latency.h"


// struct AlgTime HafTime;
namespace hozon {
namespace netaos {
namespace adf {
// basic type typedef
using float32_t = float;
using float64_t = double;
using float128_t = long double;

struct Header {
    uint64_t seq;
    uint64_t timestamp_real_us;
    uint64_t timestamp_virt_us;
    LatencyInformation latency_info;
};

struct BaseData {
    Header __header;
    std::shared_ptr<google::protobuf::Message> proto_msg = nullptr;
    std::shared_ptr<IDLBaseType> idl_msg = nullptr;
};

using BaseDataTypePtr = std::shared_ptr<BaseData>;

/* ******************************************************************************
    结构 名        :  Header
    功能描述       :  含有时间戳信息的结构体
****************************************************************************** */
struct AlgHeader {
    uint32_t seq;
    std::string frame_id;
    struct AlgTime time_stamp;
    struct AlgTime gnss_stamp;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon