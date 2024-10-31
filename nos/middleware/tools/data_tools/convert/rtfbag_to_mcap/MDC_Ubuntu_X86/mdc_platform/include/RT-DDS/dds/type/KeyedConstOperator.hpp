/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDCONSTOPERATOR_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDCONSTOPERATOR_HPP

#include <cstdint>
#include <string>

namespace dds {
namespace type {
/**
 * @brief A class represent the base of all type member operator that only need to read the member
 * @req{AR-iAOS-RCS-DDS-01004,
 * DCPS shall identify the keyed or no-keyed data types,
 * QM,
 * DR-iAOS3-RCS-DDS-00153
 * }
 */
class KeyedConstOperator {
public:
    KeyedConstOperator() = default;
    KeyedConstOperator(const KeyedConstOperator& rhs) = delete;
    KeyedConstOperator(KeyedConstOperator&& rhs) noexcept = delete;
    KeyedConstOperator& operator=(const KeyedConstOperator& rhs) = delete;
    KeyedConstOperator& operator=(KeyedConstOperator&& rhs) noexcept = delete;
    virtual ~KeyedConstOperator() = default;

    virtual bool operator()(const uint32_t& i) = 0;

    virtual bool operator()(const uint64_t& i) = 0;

    virtual bool operator()(const std::string& str) = 0;
};
}
}

#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDCONSTOPERATOR_HPP
