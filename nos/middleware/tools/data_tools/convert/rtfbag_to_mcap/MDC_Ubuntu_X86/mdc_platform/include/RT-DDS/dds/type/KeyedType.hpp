/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDTYPE_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDTYPE_HPP

#include "KeyedConstOperator.hpp"
#include "KeyedModifiableOperator.hpp"

namespace dds {
namespace type {
/**
 * @brief The base class of all data types that have keys, which is used for the communication of instance
 * @req{AR-iAOS-RCS-DDS-01004,
 * DCPS shall identify the keyed or no-keyed data types,
 * QM,
 * DR-iAOS3-RCS-DDS-00153
 * }
 */
class KeyedType {
public:
    KeyedType() = default;

    virtual bool IterateKey(KeyedModifiableOperator& modifiableOperator) = 0;

    virtual bool IterateKey(KeyedConstOperator& constOperator) const = 0;

    virtual ~KeyedType() = default;
};
}
}

#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_TYPE_KEYEDTYPE_HPP
