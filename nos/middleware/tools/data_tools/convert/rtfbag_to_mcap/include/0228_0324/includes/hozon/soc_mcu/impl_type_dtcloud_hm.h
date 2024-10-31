/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HM_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_hmtotms.h"
#include "hozon/soc_mcu/impl_type_dtcloud_tmstohm.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HM {
    ::hozon::soc_mcu::DtCloud_HmToTms HM_HmToTms;
    ::hozon::soc_mcu::DtCloud_TmsToHm HM_TmsToHm;
    ::UInt8 HM_Callback_Couter;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(HM_HmToTms);
        fun(HM_TmsToHm);
        fun(HM_Callback_Couter);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(HM_HmToTms);
        fun(HM_TmsToHm);
        fun(HM_Callback_Couter);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("HM_HmToTms", HM_HmToTms);
        fun("HM_TmsToHm", HM_TmsToHm);
        fun("HM_Callback_Couter", HM_Callback_Couter);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("HM_HmToTms", HM_HmToTms);
        fun("HM_TmsToHm", HM_TmsToHm);
        fun("HM_Callback_Couter", HM_Callback_Couter);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HM& t) const
    {
        return (HM_HmToTms == t.HM_HmToTms) && (HM_TmsToHm == t.HM_TmsToHm) && (HM_Callback_Couter == t.HM_Callback_Couter);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HM_H
