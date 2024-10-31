/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: HistoryKind.hpp
 */

#ifndef DDS_CORE_POLICY_HISTORY_KIND_HPP
#define DDS_CORE_POLICY_HISTORY_KIND_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
enum class HistoryKind : std::uint8_t {
    KEEP_LAST,
    KEEP_ALL
};
}
}
}

#endif /* DDS_CORE_POLICY_HISTORY_KIND_HPP */

