/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_LATENCYSTATUS_H
#define RTF_MAINTAIND_IMPL_TYPE_LATENCYSTATUS_H
#include "rtf/maintaind/latency/impl_type_latencystatusmap.h"

namespace rtf {
namespace maintaind {
struct LatencyStatus {
    ::rtf::maintaind::LatencyStatusMap pubStatusMap_;
    ::rtf::maintaind::LatencyStatusMap subStatusMap_;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(pubStatusMap_);
        fun(subStatusMap_);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(pubStatusMap_);
        fun(subStatusMap_);
    }

    bool operator == (const ::rtf::maintaind::LatencyStatus& t) const noexcept
    {
        return (pubStatusMap_ == t.pubStatusMap_) && (subStatusMap_ == t.subStatusMap_);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_LATENCYSTATUS_H