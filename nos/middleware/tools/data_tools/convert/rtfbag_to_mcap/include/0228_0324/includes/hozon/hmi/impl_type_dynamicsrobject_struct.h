/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_DYNAMICSROBJECT_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_DYNAMICSROBJECT_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_uint8_t.h"
#include "hozon/hmi/impl_type_poscoordlocal_struct.h"
#include "hozon/hmi/impl_type_obsize_struct.h"

namespace hozon {
namespace hmi {
struct DynamicSRObject_Struct {
    ::uint32_t Id;
    ::uint8_t Type;
    ::uint8_t BrakeLightStatus;
    ::uint8_t CarLightStatus;
    ::uint8_t Padding_u8_1;
    ::hozon::hmi::PosCoordLocal_Struct LocalPose;
    float Heading;
    ::hozon::hmi::ObSize_Struct Obsize;
    ::uint32_t IsHightLight;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Id);
        fun(Type);
        fun(BrakeLightStatus);
        fun(CarLightStatus);
        fun(Padding_u8_1);
        fun(LocalPose);
        fun(Heading);
        fun(Obsize);
        fun(IsHightLight);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Id);
        fun(Type);
        fun(BrakeLightStatus);
        fun(CarLightStatus);
        fun(Padding_u8_1);
        fun(LocalPose);
        fun(Heading);
        fun(Obsize);
        fun(IsHightLight);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Id", Id);
        fun("Type", Type);
        fun("BrakeLightStatus", BrakeLightStatus);
        fun("CarLightStatus", CarLightStatus);
        fun("Padding_u8_1", Padding_u8_1);
        fun("LocalPose", LocalPose);
        fun("Heading", Heading);
        fun("Obsize", Obsize);
        fun("IsHightLight", IsHightLight);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Id", Id);
        fun("Type", Type);
        fun("BrakeLightStatus", BrakeLightStatus);
        fun("CarLightStatus", CarLightStatus);
        fun("Padding_u8_1", Padding_u8_1);
        fun("LocalPose", LocalPose);
        fun("Heading", Heading);
        fun("Obsize", Obsize);
        fun("IsHightLight", IsHightLight);
    }

    bool operator==(const ::hozon::hmi::DynamicSRObject_Struct& t) const
    {
        return (Id == t.Id) && (Type == t.Type) && (BrakeLightStatus == t.BrakeLightStatus) && (CarLightStatus == t.CarLightStatus) && (Padding_u8_1 == t.Padding_u8_1) && (LocalPose == t.LocalPose) && (fabs(static_cast<double>(Heading - t.Heading)) < DBL_EPSILON) && (Obsize == t.Obsize) && (IsHightLight == t.IsHightLight);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_DYNAMICSROBJECT_STRUCT_H
