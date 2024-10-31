/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef DDS_CORE_POLICY_DATA_PROCESS_TYPE_KIND_HPP
#define DDS_CORE_POLICY_DATA_PROCESS_TYPE_KIND_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
 * @defgroup DIRECT_RETURN DIRECT_RETURN_POLICY
 * @brief Specify whether the user wish to let dds directly return data using OnDataProcess
 * or use the standard OnDataAvailable which is the default option
 * @ingroup DIRECT_RETURN
 */
enum class DataProcessTypeKind : std::uint8_t {
    DIRECT_DATA_PROCESS,
    NORMAL_TAKE
};
}
}
}

#endif // DDS_CORE_POLICY_DATA_PROCESS_TYPE_KIND_HPP
