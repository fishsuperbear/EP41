/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDMODIFIABLEOPERATOR_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDMODIFIABLEOPERATOR_HPP

#include <cstdint>
#include <string>

namespace dds {
namespace type {
/**
 * @brief A class represent the base of all type member operator that would modify the member
 * @req{AR-iAOS-RCS-DDS-01004,
 * DCPS shall identify the keyed or no-keyed data types,
 * QM,
 * DR-iAOS3-RCS-DDS-00153
 * }
 */
class KeyedModifiableOperator {
public:
    KeyedModifiableOperator() = default;
    KeyedModifiableOperator(const KeyedModifiableOperator& rhs) = delete;
    KeyedModifiableOperator(KeyedModifiableOperator&& rhs) noexcept = delete;
    KeyedModifiableOperator& operator=(const KeyedModifiableOperator& rhs) = delete;
    KeyedModifiableOperator& operator=(KeyedModifiableOperator&& rhs) noexcept = delete;
    virtual ~KeyedModifiableOperator() = default;

    virtual bool operator()(uint32_t& i) = 0;

    virtual bool operator()(uint64_t& i) = 0;

    virtual bool operator()(std::string& str) = 0;
};
}
}

#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDMODIFIABLEOPERATOR_HPP
