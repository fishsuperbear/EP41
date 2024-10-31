/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFFLTMGREVENTBUSOUTARRAY_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFFLTMGREVENTBUSOUTARRAY_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtdebug_hafsocclustfltvalinfo.h"
#include "hozon/soc_mcu/impl_type_dtdebug_hafcomdatafltinfo.h"
#include "hozon/soc_mcu/impl_type_dtdebug_hafpartnrecusysfltinfo.h"
#include "hozon/soc_mcu/impl_type_dtdebug_hafplathealthmngrinfo.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_HafFltMgrEventBusOutArray {
    ::UInt8 isFltMgrValidSt;
    ::hozon::soc_mcu::DtDebug_HafsocClustFltValInfo socClustFltValInfo;
    ::hozon::soc_mcu::DtDebug_HafcomDataFltInfo comDataFltInfo;
    ::hozon::soc_mcu::DtDebug_HafpartnrEcuSysFltInfo partnrEcuSysFltInfo;
    ::hozon::soc_mcu::DtDebug_HafplatHealthMngrInfo platHealthMngrInfo;
    ::UInt8 mcuPlatFltValInfo;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isFltMgrValidSt);
        fun(socClustFltValInfo);
        fun(comDataFltInfo);
        fun(partnrEcuSysFltInfo);
        fun(platHealthMngrInfo);
        fun(mcuPlatFltValInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isFltMgrValidSt);
        fun(socClustFltValInfo);
        fun(comDataFltInfo);
        fun(partnrEcuSysFltInfo);
        fun(platHealthMngrInfo);
        fun(mcuPlatFltValInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("isFltMgrValidSt", isFltMgrValidSt);
        fun("socClustFltValInfo", socClustFltValInfo);
        fun("comDataFltInfo", comDataFltInfo);
        fun("partnrEcuSysFltInfo", partnrEcuSysFltInfo);
        fun("platHealthMngrInfo", platHealthMngrInfo);
        fun("mcuPlatFltValInfo", mcuPlatFltValInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("isFltMgrValidSt", isFltMgrValidSt);
        fun("socClustFltValInfo", socClustFltValInfo);
        fun("comDataFltInfo", comDataFltInfo);
        fun("partnrEcuSysFltInfo", partnrEcuSysFltInfo);
        fun("platHealthMngrInfo", platHealthMngrInfo);
        fun("mcuPlatFltValInfo", mcuPlatFltValInfo);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_HafFltMgrEventBusOutArray& t) const
    {
        return (isFltMgrValidSt == t.isFltMgrValidSt) && (socClustFltValInfo == t.socClustFltValInfo) && (comDataFltInfo == t.comDataFltInfo) && (partnrEcuSysFltInfo == t.partnrEcuSysFltInfo) && (platHealthMngrInfo == t.platHealthMngrInfo) && (mcuPlatFltValInfo == t.mcuPlatFltValInfo);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFFLTMGREVENTBUSOUTARRAY_H
