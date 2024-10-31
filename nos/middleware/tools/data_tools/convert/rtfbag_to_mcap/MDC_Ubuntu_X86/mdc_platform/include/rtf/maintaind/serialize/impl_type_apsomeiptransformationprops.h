/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_APSOMEIPTRANSFORMATIONPROPS_H
#define RTF_MAINTAIND_IMPL_TYPE_APSOMEIPTRANSFORMATIONPROPS_H
#include "rtf/stdtype/impl_type_uint8_t.h"
#include "rtf/maintaind/serialize/impl_type_byteorder.h"
#include "rtf/maintaind/serialize/impl_type_wiretype.h"

namespace rtf {
namespace maintaind {
struct ApSomeipTransformationProps {
    ::rtf::stdtype::uint8_t alignment {1U};
    ::rtf::maintaind::ByteOrder byteorder {::rtf::maintaind::ByteOrder::BIGENDIAN};
    bool implementsLegencyStringSerialization {false};
    ::rtf::maintaind::WireType wireType {::rtf::maintaind:: WireType::STATIC};
    ::rtf::stdtype::uint8_t arrayLengthField {0U};
    ::rtf::stdtype::uint8_t mapLengthField {4U};
    ::rtf::stdtype::uint8_t vectorLengthField {4U};
    ::rtf::stdtype::uint8_t stringLengthField {4U};
    ::rtf::stdtype::uint8_t structLengthField {0U};
    ::rtf::stdtype::uint8_t unionLengthField {4U};
    ::rtf::stdtype::uint8_t unionTypeSelectorLength {4U};

    static bool IsPlane() noexcept { return true; }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(alignment);
        fun(byteorder);
        fun(implementsLegencyStringSerialization);
        fun(wireType);
        fun(arrayLengthField);
        fun(mapLengthField);
        fun(vectorLengthField);
        fun(stringLengthField);
        fun(structLengthField);
        fun(unionLengthField);
        fun(unionTypeSelectorLength);
    }

    bool operator==(const ::rtf::maintaind::ApSomeipTransformationProps& t) const
    {
        return (alignment == t.alignment) && (byteorder == t.byteorder) &&
        (implementsLegencyStringSerialization == t.implementsLegencyStringSerialization) && (wireType == t.wireType) &&
        (arrayLengthField == t.arrayLengthField) && (mapLengthField == t.mapLengthField) &&
        (vectorLengthField == t.vectorLengthField) && (stringLengthField == t.stringLengthField) &&
        (structLengthField == t.structLengthField) && (unionLengthField == t.unionLengthField) &&
        (unionTypeSelectorLength == t.unionTypeSelectorLength);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_APSOMEIPTRANSFORMATIONPROPS_H
