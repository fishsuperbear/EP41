/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_METHODINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_METHODINFO_H
#include "rtf/maintaind/impl_type_ddsmethodinfo.h"
#include "rtf/maintaind/impl_type_someipmethodinfo.h"

namespace rtf {
namespace maintaind {
struct MethodInfo {
    ::rtf::maintaind::DDSMethodInfo ddsMethodInfo_;
    ::rtf::maintaind::SomeipMethodInfo someipMethodInfo_;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(ddsMethodInfo_);
        fun(someipMethodInfo_);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(ddsMethodInfo_);
        fun(someipMethodInfo_);
    }

    bool operator == (const ::rtf::maintaind::MethodInfo& t) const noexcept
    {
        return (ddsMethodInfo_ == t.ddsMethodInfo_) && (someipMethodInfo_ == t.someipMethodInfo_);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_METHODINFO_H
