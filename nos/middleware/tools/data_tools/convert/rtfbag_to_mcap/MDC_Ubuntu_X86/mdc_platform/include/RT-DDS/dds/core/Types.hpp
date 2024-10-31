/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Types.hpp
 */

#ifndef DDS_CORE_TYPES_HPP
#define DDS_CORE_TYPES_HPP

#include <vector>
#include <string>

namespace dds {
namespace core {
using OctetSeq = std::vector<uint8_t>;

using StringSeq = std::vector<std::string>;

enum class CacheStatus : uint8_t {
    EMPTY,
    NORMAL,
    FULL
};

enum class StatisticKind : uint8_t {
    RECV_PACKS,
    DISCARD_PACKS,
    READ_BY_USER,
    SEND_PACKS,
    DISCARD_BY_SENDER,
    LATENCY_AVG,
    LATENCY_MAX,
    LATENCY_MAX_TIMESTAMP,
};

}
}

#endif /* DDS_CORE_TYPES_HPP */

