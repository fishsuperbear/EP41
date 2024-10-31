/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_VEHICLEINFO_H
#define ARA_VEHICLE_IMPL_TYPE_VEHICLEINFO_H
#include <cfloat>
#include <cmath>
#include "ara/vehicle/impl_type_time.h"
#include "ara/vehicle/impl_type_swsmfr01info.h"
#include "ara/vehicle/impl_type_vcufr06info.h"
#include "ara/vehicle/impl_type_egsmfr01info.h"
#include "ara/vehicle/impl_type_bdmfr01info.h"
#include "ara/vehicle/impl_type_acufr01info.h"
#include "ara/vehicle/impl_type_escfr07info.h"
#include "ara/vehicle/impl_type_escfr01info.h"
#include "ara/vehicle/impl_type_epsfr03info.h"
#include "ara/vehicle/impl_type_epsfr02info.h"
#include "ara/vehicle/impl_type_vcufr0finfo.h"
#include "ara/vehicle/impl_type_ehbfr01info.h"
#include "ara/vehicle/impl_type_escfr05info.h"
#include "ara/vehicle/impl_type_escfr06info.h"
#include "ara/vehicle/impl_type_escfr08info.h"
#include "ara/vehicle/impl_type_vcufr05info.h"

namespace ara {
namespace vehicle {
struct VehicleInfo {
    ::ara::vehicle::Time time;
    ::ara::vehicle::SWSMFr01Info swsm_fr01_info;
    ::ara::vehicle::VCUFr06Info vcu_fr06_info;
    ::ara::vehicle::EGSMFr01Info egsm_fr01_info;
    ::ara::vehicle::BDMFr01Info bdm_fr01_info;
    ::ara::vehicle::ACUFr01Info acu_fr01_info;
    ::ara::vehicle::ESCFr07Info esc_fr07_info;
    ::ara::vehicle::ESCFr01Info esc_fr01_info;
    ::ara::vehicle::EPSFr03Info eps_fr03_info;
    ::ara::vehicle::EPSFr02Info eps_fr02_info;
    ::ara::vehicle::VCUFr0fInfo vcu_fr0f_info;
    ::ara::vehicle::EHBFr01Info ehb_fr01_info;
    ::ara::vehicle::ESCFr05Info esc_fr05_info;
    ::ara::vehicle::ESCFr06Info esc_fr06_info;
    ::ara::vehicle::ESCFr08Info esc_fr08_info;
    ::ara::vehicle::VCUFr05Info vcu_fr05_info;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time);
        fun(swsm_fr01_info);
        fun(vcu_fr06_info);
        fun(egsm_fr01_info);
        fun(bdm_fr01_info);
        fun(acu_fr01_info);
        fun(esc_fr07_info);
        fun(esc_fr01_info);
        fun(eps_fr03_info);
        fun(eps_fr02_info);
        fun(vcu_fr0f_info);
        fun(ehb_fr01_info);
        fun(esc_fr05_info);
        fun(esc_fr06_info);
        fun(esc_fr08_info);
        fun(vcu_fr05_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time);
        fun(swsm_fr01_info);
        fun(vcu_fr06_info);
        fun(egsm_fr01_info);
        fun(bdm_fr01_info);
        fun(acu_fr01_info);
        fun(esc_fr07_info);
        fun(esc_fr01_info);
        fun(eps_fr03_info);
        fun(eps_fr02_info);
        fun(vcu_fr0f_info);
        fun(ehb_fr01_info);
        fun(esc_fr05_info);
        fun(esc_fr06_info);
        fun(esc_fr08_info);
        fun(vcu_fr05_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time", time);
        fun("swsm_fr01_info", swsm_fr01_info);
        fun("vcu_fr06_info", vcu_fr06_info);
        fun("egsm_fr01_info", egsm_fr01_info);
        fun("bdm_fr01_info", bdm_fr01_info);
        fun("acu_fr01_info", acu_fr01_info);
        fun("esc_fr07_info", esc_fr07_info);
        fun("esc_fr01_info", esc_fr01_info);
        fun("eps_fr03_info", eps_fr03_info);
        fun("eps_fr02_info", eps_fr02_info);
        fun("vcu_fr0f_info", vcu_fr0f_info);
        fun("ehb_fr01_info", ehb_fr01_info);
        fun("esc_fr05_info", esc_fr05_info);
        fun("esc_fr06_info", esc_fr06_info);
        fun("esc_fr08_info", esc_fr08_info);
        fun("vcu_fr05_info", vcu_fr05_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time", time);
        fun("swsm_fr01_info", swsm_fr01_info);
        fun("vcu_fr06_info", vcu_fr06_info);
        fun("egsm_fr01_info", egsm_fr01_info);
        fun("bdm_fr01_info", bdm_fr01_info);
        fun("acu_fr01_info", acu_fr01_info);
        fun("esc_fr07_info", esc_fr07_info);
        fun("esc_fr01_info", esc_fr01_info);
        fun("eps_fr03_info", eps_fr03_info);
        fun("eps_fr02_info", eps_fr02_info);
        fun("vcu_fr0f_info", vcu_fr0f_info);
        fun("ehb_fr01_info", ehb_fr01_info);
        fun("esc_fr05_info", esc_fr05_info);
        fun("esc_fr06_info", esc_fr06_info);
        fun("esc_fr08_info", esc_fr08_info);
        fun("vcu_fr05_info", vcu_fr05_info);
    }

    bool operator==(const ::ara::vehicle::VehicleInfo& t) const
    {
        return (time == t.time) && (swsm_fr01_info == t.swsm_fr01_info) && (vcu_fr06_info == t.vcu_fr06_info) && (egsm_fr01_info == t.egsm_fr01_info) && (bdm_fr01_info == t.bdm_fr01_info) && (acu_fr01_info == t.acu_fr01_info) && (esc_fr07_info == t.esc_fr07_info) && (esc_fr01_info == t.esc_fr01_info) && (eps_fr03_info == t.eps_fr03_info) && (eps_fr02_info == t.eps_fr02_info) && (vcu_fr0f_info == t.vcu_fr0f_info) && (ehb_fr01_info == t.ehb_fr01_info) && (esc_fr05_info == t.esc_fr05_info) && (esc_fr06_info == t.esc_fr06_info) && (esc_fr08_info == t.esc_fr08_info) && (vcu_fr05_info == t.vcu_fr05_info);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_VEHICLEINFO_H
