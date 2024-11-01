/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_APPREGISTERINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_APPREGISTERINFO_H
#include "rtf/stdtype/impl_type_uint8_t.h"
#include "rtf/maintaind/impl_type_eventregisterinfo.h"
#include "rtf/maintaind/impl_type_methodregisterinfo.h"
#include "rtf/maintaind/impl_type_fieldregisterinfo.h"
#include "rtf/maintaind/impl_type_nodecommonregisterinfo.h"
#include "rtf/maintaind/latency/impl_type_latencyinfo.h"
#include "rtf/maintaind/impl_type_listenermap.h"
namespace rtf {
namespace maintaind {
enum class AppRegisterInfoType : rtf::stdtype::uint8_t {
    EVENT_INFO,
    METHOD_INFO,
    FIELD_INFO,
    NODE_INFO,
    LATENCY_INFO,
    LISTENER_INFO
};
struct AppRegisterInfo {
    bool isPub;
    AppRegisterInfoType infoType;
    EventRegisterInfo eventInfo;
    MethodRegisterInfo methodInfo;
    FieldRegisterInfo fieldInfo;
    NodeCommonRegisterInfo nodeInfo;
    LatencyInfo latencyInfo;
    ListenerMap listenerInfo;


    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(isPub);
        fun(infoType);
        fun(eventInfo);
        fun(methodInfo);
        fun(fieldInfo);
        fun(nodeInfo);
        fun(latencyInfo);
        fun(listenerInfo);
    }

    template<typename F>
    void enumerate(F& fun) const noexcept
    {
        fun(isPub);
        fun(infoType);
        fun(eventInfo);
        fun(methodInfo);
        fun(fieldInfo);
        fun(nodeInfo);
        fun(latencyInfo);
        fun(listenerInfo);
    }

    bool operator == (const ::rtf::maintaind::AppRegisterInfo& t) const noexcept
    {
        return (isPub == t.isPub) && (infoType == t.infoType) && (eventInfo == t.eventInfo) &&
            (methodInfo == t.methodInfo) && (fieldInfo == t.fieldInfo) && (nodeInfo == t.nodeInfo) &&
            (latencyInfo == t.latencyInfo) && (listenerInfo == t.listenerInfo);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_APPREGISTERINFO_H
