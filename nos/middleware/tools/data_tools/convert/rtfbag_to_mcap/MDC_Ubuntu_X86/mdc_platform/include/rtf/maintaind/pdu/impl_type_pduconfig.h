/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This provide the data structure to store pdu info.
 * Create: 2022-03-29
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_PDUCONFIG_H
#define RTF_MAINTAIND_IMPL_TYPE_PDUCONFIG_H
#include "rtf/stdtype/impl_type_string.h"
#include "rtf/stdtype/impl_type_uint64_t.h"
#include "rtf/stdtype/impl_type_boolean.h"
#include "rtf/maintaind/pdu/impl_type_isignalinfovector.h"
namespace rtf {
namespace maintaind {
struct PduConfigInfo {
    ::rtf::stdtype::String pduName_;
    ::rtf::stdtype::uint64_t length_;
    ::rtf::stdtype::boolean isDynamic_;
    ::rtf::maintaind::IsignalInfoVector isignalInfoList_;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(pduName_);
        fun(length_);
        fun(isDynamic_);
        fun(isignalInfoList_);
    }

    bool operator == (const ::rtf::maintaind::PduConfigInfo& t) const noexcept
    {
        return (pduName_ == t.pduName_) && (length_ == t.length_) &&
        (isDynamic_ == t.isDynamic_) && (isignalInfoList_ == t.isignalInfoList_);
    }
};
} // namespace maintaind
} // namespace rtf

#endif // RTF_MAINTAIND_IMPL_TYPE_PDUCONFIG_H
