/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_SOMEIPMETHODINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_SOMEIPMETHODINFO_H
#include "rtf/stdtype/impl_type_uint16_t.h"
#include "rtf/stdtype/impl_type_string.h"
#include "rtf/stdtype/impl_type_boolean.h"
#include "rtf/maintaind/e2e/impl_type_e2econfiginfo.h"
#include "rtf/maintaind/serialize/impl_type_serializeconfigrawdata.h"
namespace rtf {
namespace maintaind {
struct SomeipMethodInfo {
    ::rtf::stdtype::uint16_t methodId_;
    ::rtf::stdtype::String instanceShortName_;
    ::rtf::stdtype::uint32_t minorVersion_;
    ::rtf::stdtype::uint16_t majorVersion_;
    ::rtf::stdtype::boolean isReliable_;
    ::rtf::maintaind::E2EConfigInfo e2eConfig_;
    ::rtf::maintaind::SerializeConfigRawData requestSerializeConfig_;
    ::rtf::maintaind::SerializeConfigRawData replySerializeConfig_;
    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(methodId_);
        fun(instanceShortName_);
        fun(minorVersion_);
        fun(majorVersion_);
        fun(isReliable_);
        fun(e2eConfig_);
        fun(requestSerializeConfig_);
        fun(replySerializeConfig_);
    }

    bool operator == (const ::rtf::maintaind::SomeipMethodInfo& t) const noexcept
    {
        return (methodId_ == t.methodId_) && (instanceShortName_ == t.instanceShortName_) &&
            (minorVersion_ == t.minorVersion_) && (majorVersion_ == t.majorVersion_) &&
            (isReliable_ == t.isReliable_) && (e2eConfig_ == t.e2eConfig_) &&
            (requestSerializeConfig_ == t.requestSerializeConfig_) &&
            (replySerializeConfig_ == t.replySerializeConfig_);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_SOMEIPMETHODINFO_H