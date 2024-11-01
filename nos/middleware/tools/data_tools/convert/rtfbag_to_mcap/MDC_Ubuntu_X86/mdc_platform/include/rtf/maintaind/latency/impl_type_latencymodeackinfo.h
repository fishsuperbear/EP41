/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_LATENCYMODEACKINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_LATENCYMODEACKINFO_H
#include "rtf/stdtype/impl_type_boolean.h"
#include "rtf/maintaind/latency/impl_type_latencymode.h"
namespace rtf {
namespace maintaind {
struct LatencyModeAckInfo {
    ::rtf::maintaind::LatencyMode latencyMode_;
    ::rtf::stdtype::boolean hasPub_;
    ::rtf::stdtype::boolean hasSub_;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(latencyMode_);
        fun(hasPub_);
        fun(hasSub_);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(latencyMode_);
        fun(hasPub_);
        fun(hasSub_);
    }

    bool operator == (const ::rtf::maintaind::LatencyModeAckInfo& t) const noexcept
    {
        return (latencyMode_ == t.latencyMode_) && (hasPub_ == t.hasPub_) && (hasSub_ == t.hasSub_);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_LATENCYMODEACKINFO_H
