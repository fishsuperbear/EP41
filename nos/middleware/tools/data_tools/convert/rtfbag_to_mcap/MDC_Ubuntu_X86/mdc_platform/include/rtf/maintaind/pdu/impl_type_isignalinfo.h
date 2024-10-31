/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This provide the data structure to store isignal info.
 * Create: 2022-03-30
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_ISIGNAL_H
#define RTF_MAINTAIND_IMPL_TYPE_ISIGNAL_H
#include "rtf/stdtype/impl_type_string.h"
#include "rtf/stdtype/impl_type_uint64_t.h"
#include "rtf/stdtype/impl_type_boolean.h"
namespace rtf {
namespace maintaind {
struct IsignalInfo {
    ::rtf::stdtype::String isignalName_;
    ::rtf::stdtype::String type_;
    ::rtf::stdtype::boolean isDynamic_;
    ::rtf::stdtype::uint64_t startPos_;
    ::rtf::stdtype::uint64_t length_;
    ::rtf::stdtype::String byteOrder_;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(isignalName_);
        fun(type_);
        fun(isDynamic_);
        fun(startPos_);
        fun(length_);
        fun(byteOrder_);
    }

    bool operator == (const ::rtf::maintaind::IsignalInfo& t) const noexcept
    {
        return (isignalName_ == t.isignalName_) && (type_ == t.type_) && (isDynamic_ == t.isDynamic_) &&
        (startPos_ == t.startPos_) && (length_ == t.length_) && (byteOrder_ == t.byteOrder_);
    }
};
} // namespace maintaind
} // namespace rtf

#endif // RTF_MAINTAIND_IMPL_TYPE_ISIGNAL_H
