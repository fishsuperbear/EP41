/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_LISTENERPARAMS_H
#define RTF_MAINTAIND_IMPL_TYPE_LISTENERPARAMS_H
#include "rtf/stdtype/impl_type_string.h"
#include "rtf/stdtype/impl_type_uint32_t.h"

namespace rtf {
namespace maintaind {
struct ListenerParams {
    ::rtf::stdtype::String   guid_;
    ::rtf::stdtype::uint32_t pubPid_;
    ::rtf::stdtype::uint32_t subPid_;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(guid_);
        fun(pubPid_);
        fun(subPid_);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(guid_);
        fun(pubPid_);
        fun(subPid_);
    }

    bool operator==(const ::rtf::maintaind::ListenerParams& t) const
    {
        return (guid_ == t.guid_) && (pubPid_ == t.pubPid_) && (subPid_ == t.subPid_);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_LISTENERPARAMS_H
