/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFPLATHEALTHMNGRINFO_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFPLATHEALTHMNGRINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_hafmsgsoctomcualivefltinfo.h"
#include "hozon/soc_mcu/impl_type_dtcloud_hafprocssocalivefltinfo.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafplatHealthMngrInfo {
    ::hozon::soc_mcu::DtCloud_HafmsgSocToMcuAliveFltInfo msg_alive_fltInfo;
    ::hozon::soc_mcu::DtCloud_HafprocsSocAliveFltInfo process_alive_fltInfo;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(msg_alive_fltInfo);
        fun(process_alive_fltInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(msg_alive_fltInfo);
        fun(process_alive_fltInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("msg_alive_fltInfo", msg_alive_fltInfo);
        fun("process_alive_fltInfo", process_alive_fltInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("msg_alive_fltInfo", msg_alive_fltInfo);
        fun("process_alive_fltInfo", process_alive_fltInfo);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafplatHealthMngrInfo& t) const
    {
        return (msg_alive_fltInfo == t.msg_alive_fltInfo) && (process_alive_fltInfo == t.process_alive_fltInfo);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFPLATHEALTHMNGRINFO_H
