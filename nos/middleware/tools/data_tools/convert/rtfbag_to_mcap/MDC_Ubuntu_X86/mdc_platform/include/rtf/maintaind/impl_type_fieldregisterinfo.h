/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2019. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_FIELDREGISTERINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_FIELDREGISTERINFO_H
#include "rtf/maintaind/impl_type_fieldmap.h"
#include "rtf/stdtype/impl_type_uint16_t.h"
#include "rtf/stdtype/impl_type_string.h"
#include "rtf/stdtype/impl_type_int32_t.h"

namespace rtf {
namespace maintaind {
struct FieldRegisterInfo {
    ::rtf::stdtype::String appName_;
    ::rtf::stdtype::String fieldType_;
    ::rtf::maintaind::FieldMap fieldMap_;
    ::rtf::stdtype::uint16_t instanceId_ = 0U;
    bool isOnline_ = false;
    ::rtf::stdtype::int32_t nodePid_ = 0;
    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(appName_);
        fun(fieldType_);
        fun(fieldMap_);
        fun(instanceId_);
        fun(isOnline_);
        fun(nodePid_);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(appName_);
        fun(fieldType_);
        fun(fieldMap_);
        fun(instanceId_);
        fun(isOnline_);
        fun(nodePid_);
    }

    bool operator == (const ::rtf::maintaind::FieldRegisterInfo& t) const noexcept
    {
        return (appName_ == t.appName_) && (fieldType_ == t.fieldType_) && (fieldMap_ == t.fieldMap_) &&
        (instanceId_ == t.instanceId_) && (isOnline_ == t.isOnline_) && (nodePid_ == t.nodePid_);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_FIELDREGISTERINFO_H