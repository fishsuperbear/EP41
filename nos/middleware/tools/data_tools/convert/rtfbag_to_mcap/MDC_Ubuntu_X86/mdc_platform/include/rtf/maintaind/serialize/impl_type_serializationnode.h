/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_SERIALIZATIONNODE_H
#define RTF_MAINTAIND_IMPL_TYPE_SERIALIZATIONNODE_H
#include "rtf/maintaind/serialize/impl_type_apsomeiptransformationprops.h"
#include "rtf/stdtype/impl_type_uint16_t.h"
#include "rtf/stdtype/impl_type_uint8_t.h"
#include "ara/core/vector.h"
namespace rtf {
namespace maintaind {
struct SerializationNode {
    ::rtf::maintaind::ApSomeipTransformationProps serializationConfig;
    ::rtf::stdtype::uint16_t dataId {0xffffu};
    ::rtf::stdtype::uint8_t tlvLengthFieldSize {0u};
    bool isChildNodeEnableTlv {false};
    bool isLastSerializeNode {false};
    ara::core::Vector<rtf::maintaind::SerializationNode> childNodeList;

    static bool IsPlane() noexcept { return false; }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(serializationConfig);
        fun(dataId);
        fun(tlvLengthFieldSize);
        fun(isChildNodeEnableTlv);
        fun(isLastSerializeNode);
        fun(childNodeList);
    }

    bool operator==(const ::rtf::maintaind::SerializationNode& t) const
    {
        return (serializationConfig == t.serializationConfig) && (dataId == t.dataId) &&
        (tlvLengthFieldSize == t.tlvLengthFieldSize) && (isChildNodeEnableTlv == t.isChildNodeEnableTlv) &&
        (isLastSerializeNode == t.isLastSerializeNode) && (childNodeList == t.childNodeList);
    }
};
using SerializationNodeList = ara::core::Vector<rtf::maintaind::SerializationNode>;
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_SERIALIZATIONNODE_H
