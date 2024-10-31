/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_MCUCLOUDDATATYPE_H
#define HOZON_SOC_MCU_IMPL_TYPE_MCUCLOUDDATATYPE_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_intf_running.h"
#include "hozon/soc_mcu/impl_type_dtcloud_hm.h"
#include "hozon/soc_mcu/impl_type_dtcloud_fm.h"
#include "hozon/soc_mcu/impl_type_dtcloud_sm.h"
#include "hozon/soc_mcu/impl_type_dtcloud_nvm.h"
#include "hozon/soc_mcu/impl_type_dtcloud_eth.h"
#include "hozon/soc_mcu/impl_type_dtcloud_os.h"
#include "hozon/soc_mcu/impl_type_dtpwronoffdata.h"
#include "hozon/soc_mcu/impl_type_dtcloud_adas.h"

namespace hozon {
namespace soc_mcu {
struct MCUCloudDataType {
    ::hozon::soc_mcu::DtCloud_INTF_Running INTFData;
    ::hozon::soc_mcu::DtCloud_HM HMData;
    ::hozon::soc_mcu::DtCloud_FM FMData;
    ::hozon::soc_mcu::DtCloud_SM SMData;
    ::hozon::soc_mcu::DtCloud_NVM NVMData;
    ::hozon::soc_mcu::DtCloud_ETH ETHData;
    ::hozon::soc_mcu::DtCloud_OS OSData;
    ::hozon::soc_mcu::DtPwrOnOffData PwrOnOffData;
    ::hozon::soc_mcu::DtCloud_ADAS ADASData;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTFData);
        fun(HMData);
        fun(FMData);
        fun(SMData);
        fun(NVMData);
        fun(ETHData);
        fun(OSData);
        fun(PwrOnOffData);
        fun(ADASData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTFData);
        fun(HMData);
        fun(FMData);
        fun(SMData);
        fun(NVMData);
        fun(ETHData);
        fun(OSData);
        fun(PwrOnOffData);
        fun(ADASData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTFData", INTFData);
        fun("HMData", HMData);
        fun("FMData", FMData);
        fun("SMData", SMData);
        fun("NVMData", NVMData);
        fun("ETHData", ETHData);
        fun("OSData", OSData);
        fun("PwrOnOffData", PwrOnOffData);
        fun("ADASData", ADASData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTFData", INTFData);
        fun("HMData", HMData);
        fun("FMData", FMData);
        fun("SMData", SMData);
        fun("NVMData", NVMData);
        fun("ETHData", ETHData);
        fun("OSData", OSData);
        fun("PwrOnOffData", PwrOnOffData);
        fun("ADASData", ADASData);
    }

    bool operator==(const ::hozon::soc_mcu::MCUCloudDataType& t) const
    {
        return (INTFData == t.INTFData) && (HMData == t.HMData) && (FMData == t.FMData) && (SMData == t.SMData) && (NVMData == t.NVMData) && (ETHData == t.ETHData) && (OSData == t.OSData) && (PwrOnOffData == t.PwrOnOffData) && (ADASData == t.ADASData);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_MCUCLOUDDATATYPE_H
