/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPDATAFAILED_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPDATAFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafLidarPercpDataFailed {
    ::UInt8 data_l_comloss_error;
    ::UInt8 data_l_vldt_error;
    ::UInt8 data_r_comloss_error;
    ::UInt8 data_r_vldt_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(data_l_comloss_error);
        fun(data_l_vldt_error);
        fun(data_r_comloss_error);
        fun(data_r_vldt_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(data_l_comloss_error);
        fun(data_l_vldt_error);
        fun(data_r_comloss_error);
        fun(data_r_vldt_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("data_l_comloss_error", data_l_comloss_error);
        fun("data_l_vldt_error", data_l_vldt_error);
        fun("data_r_comloss_error", data_r_comloss_error);
        fun("data_r_vldt_error", data_r_vldt_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("data_l_comloss_error", data_l_comloss_error);
        fun("data_l_vldt_error", data_l_vldt_error);
        fun("data_r_comloss_error", data_r_comloss_error);
        fun("data_r_vldt_error", data_r_vldt_error);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafLidarPercpDataFailed& t) const
    {
        return (data_l_comloss_error == t.data_l_comloss_error) && (data_l_vldt_error == t.data_l_vldt_error) && (data_r_comloss_error == t.data_r_comloss_error) && (data_r_vldt_error == t.data_r_vldt_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPDATAFAILED_H
