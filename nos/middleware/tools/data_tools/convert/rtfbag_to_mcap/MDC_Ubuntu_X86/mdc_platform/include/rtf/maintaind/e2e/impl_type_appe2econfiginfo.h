/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_APPE2ECONFIGINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_APPE2ECONFIGINFO_H
#include "rtf/maintaind/e2e/impl_type_evente2econfiginfovector.h"

namespace rtf {
namespace maintaind {
struct AppE2EConfigInfo {
    ::rtf::maintaind::EventE2EConfigInfoVector eventE2EConfigList;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(eventE2EConfigList);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(eventE2EConfigList);
    }

    bool operator == (const ::rtf::maintaind::AppE2EConfigInfo& t) const noexcept
    {
        return (eventE2EConfigList == t.eventE2EConfigList);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_APPE2ECONFIGINFO_H