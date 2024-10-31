/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_EHBFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_EHBFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct EHBFr01Info {
    ::UInt8 brake_pedal_appliedv;
    ::UInt8 brake_pedal_applied;
    ::UInt8 ehb_fr01_msgcounter;
    ::UInt8 ehb_fr01_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(brake_pedal_appliedv);
        fun(brake_pedal_applied);
        fun(ehb_fr01_msgcounter);
        fun(ehb_fr01_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(brake_pedal_appliedv);
        fun(brake_pedal_applied);
        fun(ehb_fr01_msgcounter);
        fun(ehb_fr01_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("brake_pedal_appliedv", brake_pedal_appliedv);
        fun("brake_pedal_applied", brake_pedal_applied);
        fun("ehb_fr01_msgcounter", ehb_fr01_msgcounter);
        fun("ehb_fr01_checksum", ehb_fr01_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("brake_pedal_appliedv", brake_pedal_appliedv);
        fun("brake_pedal_applied", brake_pedal_applied);
        fun("ehb_fr01_msgcounter", ehb_fr01_msgcounter);
        fun("ehb_fr01_checksum", ehb_fr01_checksum);
    }

    bool operator==(const ::ara::vehicle::EHBFr01Info& t) const
    {
        return (brake_pedal_appliedv == t.brake_pedal_appliedv) && (brake_pedal_applied == t.brake_pedal_applied) && (ehb_fr01_msgcounter == t.ehb_fr01_msgcounter) && (ehb_fr01_checksum == t.ehb_fr01_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_EHBFR01INFO_H
