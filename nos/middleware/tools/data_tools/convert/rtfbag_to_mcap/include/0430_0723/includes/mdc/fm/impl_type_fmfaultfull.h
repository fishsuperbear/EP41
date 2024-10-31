/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_FMFAULTFULL_H
#define MDC_FM_IMPL_TYPE_FMFAULTFULL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_string.h"
#include "impl_type_int32.h"
#include "mdc/fm/impl_type_fmstringvec.h"

namespace mdc {
namespace fm {
struct FmFaultFull {
    ::UInt16 faultId;
    ::UInt16 faultObj;
    ::UInt8 phyCpuNo;
    ::UInt8 cpuCoreNo;
    ::UInt8 type;
    ::UInt8 clss;
    ::UInt8 level;
    ::UInt32 flag;
    ::UInt32 componentId;
    ::UInt32 count;
    ::String timeFirst;
    ::String timeLast;
    ::String para;
    ::String desc;
    ::String actionName;
    ::Int32 actionResult;
    ::mdc::fm::FmStringVec subFaultVec;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(faultId);
        fun(faultObj);
        fun(phyCpuNo);
        fun(cpuCoreNo);
        fun(type);
        fun(clss);
        fun(level);
        fun(flag);
        fun(componentId);
        fun(count);
        fun(timeFirst);
        fun(timeLast);
        fun(para);
        fun(desc);
        fun(actionName);
        fun(actionResult);
        fun(subFaultVec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultId);
        fun(faultObj);
        fun(phyCpuNo);
        fun(cpuCoreNo);
        fun(type);
        fun(clss);
        fun(level);
        fun(flag);
        fun(componentId);
        fun(count);
        fun(timeFirst);
        fun(timeLast);
        fun(para);
        fun(desc);
        fun(actionName);
        fun(actionResult);
        fun(subFaultVec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("phyCpuNo", phyCpuNo);
        fun("cpuCoreNo", cpuCoreNo);
        fun("type", type);
        fun("clss", clss);
        fun("level", level);
        fun("flag", flag);
        fun("componentId", componentId);
        fun("count", count);
        fun("timeFirst", timeFirst);
        fun("timeLast", timeLast);
        fun("para", para);
        fun("desc", desc);
        fun("actionName", actionName);
        fun("actionResult", actionResult);
        fun("subFaultVec", subFaultVec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("phyCpuNo", phyCpuNo);
        fun("cpuCoreNo", cpuCoreNo);
        fun("type", type);
        fun("clss", clss);
        fun("level", level);
        fun("flag", flag);
        fun("componentId", componentId);
        fun("count", count);
        fun("timeFirst", timeFirst);
        fun("timeLast", timeLast);
        fun("para", para);
        fun("desc", desc);
        fun("actionName", actionName);
        fun("actionResult", actionResult);
        fun("subFaultVec", subFaultVec);
    }

    bool operator==(const ::mdc::fm::FmFaultFull& t) const
    {
        return (faultId == t.faultId) && (faultObj == t.faultObj) && (phyCpuNo == t.phyCpuNo) && (cpuCoreNo == t.cpuCoreNo) && (type == t.type) && (clss == t.clss) && (level == t.level) && (flag == t.flag) && (componentId == t.componentId) && (count == t.count) && (timeFirst == t.timeFirst) && (timeLast == t.timeLast) && (para == t.para) && (desc == t.desc) && (actionName == t.actionName) && (actionResult == t.actionResult) && (subFaultVec == t.subFaultVec);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_FMFAULTFULL_H
