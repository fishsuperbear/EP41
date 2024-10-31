/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SERVCALLFAIL_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SERVCALLFAIL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_ServCallFail {
    ::UInt8 ServCallFailFlg;
    ::UInt8 ServCallFailCnt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ServCallFailFlg);
        fun(ServCallFailCnt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ServCallFailFlg);
        fun(ServCallFailCnt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ServCallFailFlg", ServCallFailFlg);
        fun("ServCallFailCnt", ServCallFailCnt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ServCallFailFlg", ServCallFailFlg);
        fun("ServCallFailCnt", ServCallFailCnt);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_ServCallFail& t) const
    {
        return (ServCallFailFlg == t.ServCallFailFlg) && (ServCallFailCnt == t.ServCallFailCnt);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SERVCALLFAIL_H
