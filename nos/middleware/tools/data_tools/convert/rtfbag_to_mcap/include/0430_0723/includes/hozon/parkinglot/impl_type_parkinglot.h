/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PARKINGLOT_IMPL_TYPE_PARKINGLOT_H
#define HOZON_PARKINGLOT_IMPL_TYPE_PARKINGLOT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "hozon/parkinglot/impl_type_pspointvector.h"
#include "hozon/common/impl_type_commontime.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace parkinglot {
struct ParkingLot {
    ::UInt32 seq;
    ::UInt8 type;
    ::UInt8 status;
    ::UInt8 sensorType;
    ::hozon::parkinglot::PsPointVector ptsVRF;
    ::hozon::common::CommonTime timeCreation;
    ::Boolean isPrivatePs;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(type);
        fun(status);
        fun(sensorType);
        fun(ptsVRF);
        fun(timeCreation);
        fun(isPrivatePs);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(type);
        fun(status);
        fun(sensorType);
        fun(ptsVRF);
        fun(timeCreation);
        fun(isPrivatePs);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("type", type);
        fun("status", status);
        fun("sensorType", sensorType);
        fun("ptsVRF", ptsVRF);
        fun("timeCreation", timeCreation);
        fun("isPrivatePs", isPrivatePs);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("type", type);
        fun("status", status);
        fun("sensorType", sensorType);
        fun("ptsVRF", ptsVRF);
        fun("timeCreation", timeCreation);
        fun("isPrivatePs", isPrivatePs);
    }

    bool operator==(const ::hozon::parkinglot::ParkingLot& t) const
    {
        return (seq == t.seq) && (type == t.type) && (status == t.status) && (sensorType == t.sensorType) && (ptsVRF == t.ptsVRF) && (timeCreation == t.timeCreation) && (isPrivatePs == t.isPrivatePs);
    }
};
} // namespace parkinglot
} // namespace hozon


#endif // HOZON_PARKINGLOT_IMPL_TYPE_PARKINGLOT_H
