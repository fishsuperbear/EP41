/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_DATAVALUE_H
#define HOZON_STATEMACHINE_IMPL_TYPE_DATAVALUE_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_int32array.h"
#include "hozon/object/impl_type_float32vector.h"
#include "impl_type_uint8_t.h"
#include "impl_type_doublevector.h"
#include "hozon/mapmsg/impl_type_stringvector.h"

namespace hozon {
namespace statemachine {
struct DataValue {
    ::hozon::composite::Int32Array int_val;
    ::hozon::object::Float32Vector float_val;
    ::uint8_t reserved;
    ::DoubleVector double_val;
    ::hozon::mapmsg::stringVector str_val;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(int_val);
        fun(float_val);
        fun(reserved);
        fun(double_val);
        fun(str_val);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(int_val);
        fun(float_val);
        fun(reserved);
        fun(double_val);
        fun(str_val);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("int_val", int_val);
        fun("float_val", float_val);
        fun("reserved", reserved);
        fun("double_val", double_val);
        fun("str_val", str_val);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("int_val", int_val);
        fun("float_val", float_val);
        fun("reserved", reserved);
        fun("double_val", double_val);
        fun("str_val", str_val);
    }

    bool operator==(const ::hozon::statemachine::DataValue& t) const
    {
        return (int_val == t.int_val) && (float_val == t.float_val) && (reserved == t.reserved) && (double_val == t.double_val) && (str_val == t.str_val);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_DATAVALUE_H
