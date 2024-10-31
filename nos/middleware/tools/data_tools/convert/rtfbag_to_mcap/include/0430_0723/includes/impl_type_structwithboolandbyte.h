/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_STRUCTWITHBOOLANDBYTE_H
#define IMPL_TYPE_STRUCTWITHBOOLANDBYTE_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"

struct StructWithBoolAndByte {
    ::Boolean BoolData;
    ::UInt8 ByteData;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(BoolData);
        fun(ByteData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(BoolData);
        fun(ByteData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("BoolData", BoolData);
        fun("ByteData", ByteData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("BoolData", BoolData);
        fun("ByteData", ByteData);
    }

    bool operator==(const ::StructWithBoolAndByte& t) const
    {
        return (BoolData == t.BoolData) && (ByteData == t.ByteData);
    }
};


#endif // IMPL_TYPE_STRUCTWITHBOOLANDBYTE_H
