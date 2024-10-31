/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_STEERINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_STEERINFO_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_float32withvalid.h"

namespace ara {
namespace actcompensation {
struct SteerInfo {
    ::ara::actcompensation::Float32WithValid steerPinionAngle;
    ::ara::actcompensation::Float32WithValid steerAngle;
    ::ara::actcompensation::Float32WithValid steerAngleRate;
    ::ara::actcompensation::Float32WithValid sasSteerAngle;
    ::ara::actcompensation::Float32WithValid sasSteerAngleRate;
    ::ara::actcompensation::Float32WithValid steerTorque;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(steerPinionAngle);
        fun(steerAngle);
        fun(steerAngleRate);
        fun(sasSteerAngle);
        fun(sasSteerAngleRate);
        fun(steerTorque);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(steerPinionAngle);
        fun(steerAngle);
        fun(steerAngleRate);
        fun(sasSteerAngle);
        fun(sasSteerAngleRate);
        fun(steerTorque);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("steerPinionAngle", steerPinionAngle);
        fun("steerAngle", steerAngle);
        fun("steerAngleRate", steerAngleRate);
        fun("sasSteerAngle", sasSteerAngle);
        fun("sasSteerAngleRate", sasSteerAngleRate);
        fun("steerTorque", steerTorque);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("steerPinionAngle", steerPinionAngle);
        fun("steerAngle", steerAngle);
        fun("steerAngleRate", steerAngleRate);
        fun("sasSteerAngle", sasSteerAngle);
        fun("sasSteerAngleRate", sasSteerAngleRate);
        fun("steerTorque", steerTorque);
    }

    bool operator==(const ::ara::actcompensation::SteerInfo& t) const
    {
        return (steerPinionAngle == t.steerPinionAngle) && (steerAngle == t.steerAngle) && (steerAngleRate == t.steerAngleRate) && (sasSteerAngle == t.sasSteerAngle) && (sasSteerAngleRate == t.sasSteerAngleRate) && (steerTorque == t.steerTorque);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_STEERINFO_H
