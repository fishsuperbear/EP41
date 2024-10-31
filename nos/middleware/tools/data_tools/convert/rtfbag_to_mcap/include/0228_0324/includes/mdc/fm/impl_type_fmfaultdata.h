/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_IMPL_TYPE_FMFAULTDATA_H
#define MDC_FM_IMPL_TYPE_FMFAULTDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "mdc/fm/impl_type_fmparaarray.h"
#include "mdc/fm/impl_type_fmcharvec.h"

namespace mdc {
namespace fm {
struct FmFaultData {
    ::UInt16 faultId;
    ::UInt16 faultObj;
    ::UInt8 phyCpuNo;
    ::UInt8 cpuCoreNo;
    ::UInt32 componentId;
    ::UInt8 type;
    ::UInt32 flag;
    ::UInt8 paraNum;
    ::mdc::fm::FmParaArray para;
    ::UInt8 clss;
    ::mdc::fm::FmCharVec time;

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
        fun(componentId);
        fun(type);
        fun(flag);
        fun(paraNum);
        fun(para);
        fun(clss);
        fun(time);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(faultId);
        fun(faultObj);
        fun(phyCpuNo);
        fun(cpuCoreNo);
        fun(componentId);
        fun(type);
        fun(flag);
        fun(paraNum);
        fun(para);
        fun(clss);
        fun(time);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("phyCpuNo", phyCpuNo);
        fun("cpuCoreNo", cpuCoreNo);
        fun("componentId", componentId);
        fun("type", type);
        fun("flag", flag);
        fun("paraNum", paraNum);
        fun("para", para);
        fun("clss", clss);
        fun("time", time);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("faultId", faultId);
        fun("faultObj", faultObj);
        fun("phyCpuNo", phyCpuNo);
        fun("cpuCoreNo", cpuCoreNo);
        fun("componentId", componentId);
        fun("type", type);
        fun("flag", flag);
        fun("paraNum", paraNum);
        fun("para", para);
        fun("clss", clss);
        fun("time", time);
    }

    bool operator==(const ::mdc::fm::FmFaultData& t) const
    {
        return (faultId == t.faultId) && (faultObj == t.faultObj) && (phyCpuNo == t.phyCpuNo) && (cpuCoreNo == t.cpuCoreNo) && (componentId == t.componentId) && (type == t.type) && (flag == t.flag) && (paraNum == t.paraNum) && (para == t.para) && (clss == t.clss) && (time == t.time);
    }
};
} // namespace fm
} // namespace mdc


#endif // MDC_FM_IMPL_TYPE_FMFAULTDATA_H
