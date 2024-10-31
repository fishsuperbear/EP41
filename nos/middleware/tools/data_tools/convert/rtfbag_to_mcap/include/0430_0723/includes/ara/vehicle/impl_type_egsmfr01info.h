/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_EGSMFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_EGSMFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct EGSMFr01Info {
    ::UInt8 egsm_gearbox_position_req;
    ::UInt8 egsm_gearbox_position_req_inv;
    ::UInt8 egsm_gearbox_position_req_validity;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(egsm_gearbox_position_req);
        fun(egsm_gearbox_position_req_inv);
        fun(egsm_gearbox_position_req_validity);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(egsm_gearbox_position_req);
        fun(egsm_gearbox_position_req_inv);
        fun(egsm_gearbox_position_req_validity);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("egsm_gearbox_position_req", egsm_gearbox_position_req);
        fun("egsm_gearbox_position_req_inv", egsm_gearbox_position_req_inv);
        fun("egsm_gearbox_position_req_validity", egsm_gearbox_position_req_validity);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("egsm_gearbox_position_req", egsm_gearbox_position_req);
        fun("egsm_gearbox_position_req_inv", egsm_gearbox_position_req_inv);
        fun("egsm_gearbox_position_req_validity", egsm_gearbox_position_req_validity);
    }

    bool operator==(const ::ara::vehicle::EGSMFr01Info& t) const
    {
        return (egsm_gearbox_position_req == t.egsm_gearbox_position_req) && (egsm_gearbox_position_req_inv == t.egsm_gearbox_position_req_inv) && (egsm_gearbox_position_req_validity == t.egsm_gearbox_position_req_validity);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_EGSMFR01INFO_H
