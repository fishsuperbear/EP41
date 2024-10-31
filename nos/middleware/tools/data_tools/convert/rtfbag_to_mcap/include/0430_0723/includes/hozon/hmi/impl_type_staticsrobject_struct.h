/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_STATICSROBJECT_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_STATICSROBJECT_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint16_t.h"
#include "hozon/hmi/impl_type_poscoordlocal_struct.h"

namespace hozon {
namespace hmi {
struct StaticSRObject_Struct {
    ::uint32_t Id;
    ::uint8_t Type;
    ::uint8_t Padding_u8_1;
    ::uint16_t Padding_u16_1;
    ::hozon::hmi::PosCoordLocal_Struct LocalPose;

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
        fun(Padding_u8_1);
        fun(Padding_u16_1);
        fun(LocalPose);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Id);
        fun(Type);
        fun(Padding_u8_1);
        fun(Padding_u16_1);
        fun(LocalPose);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Id", Id);
        fun("Type", Type);
        fun("Padding_u8_1", Padding_u8_1);
        fun("Padding_u16_1", Padding_u16_1);
        fun("LocalPose", LocalPose);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Id", Id);
        fun("Type", Type);
        fun("Padding_u8_1", Padding_u8_1);
        fun("Padding_u16_1", Padding_u16_1);
        fun("LocalPose", LocalPose);
    }

    bool operator==(const ::hozon::hmi::StaticSRObject_Struct& t) const
    {
        return (Id == t.Id) && (Type == t.Type) && (Padding_u8_1 == t.Padding_u8_1) && (Padding_u16_1 == t.Padding_u16_1) && (LocalPose == t.LocalPose);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_STATICSROBJECT_STRUCT_H
