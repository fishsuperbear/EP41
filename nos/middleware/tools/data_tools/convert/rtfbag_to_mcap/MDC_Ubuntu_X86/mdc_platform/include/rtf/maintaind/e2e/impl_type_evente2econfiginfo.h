/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_EVENTE2ECONFIGINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_EVENTE2ECONFIGINFO_H
#include "rtf/stdtype/impl_type_uint16_t.h"
#include "rtf/maintaind/impl_type_drivertype.h"
#include "rtf/stdtype/impl_type_string.h"
#include "rtf/maintaind/e2e/impl_type_e2econfiginfo.h"

namespace rtf {
namespace maintaind {
struct EventE2EConfigInfo {
    ::rtf::stdtype::uint16_t serviceId;
    ::rtf::stdtype::uint16_t instanceId;
    ::rtf::maintaind::DriverType driverType;
    ::rtf::stdtype::uint16_t eventId;
    ::rtf::stdtype::String eventTopicName;
    ::rtf::maintaind::E2EConfigInfo e2eConfigInfo;

    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(serviceId);
        fun(instanceId);
        fun(driverType);
        fun(eventId);
        fun(eventTopicName);
        fun(e2eConfigInfo);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(serviceId);
        fun(instanceId);
        fun(driverType);
        fun(eventId);
        fun(eventTopicName);
        fun(e2eConfigInfo);
    }

    bool operator == (const ::rtf::maintaind::EventE2EConfigInfo& t) const noexcept
    {
        return (serviceId == t.serviceId) && (instanceId == t.instanceId) && (driverType == t.driverType) &&
               (eventId == t.eventId) && (eventTopicName == t.eventTopicName) && (e2eConfigInfo == t.e2eConfigInfo);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_EVENTE2ECONFIGINFO_H
