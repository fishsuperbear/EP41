/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HPP_PATH_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_HPP_PATH_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "hozon/hmi/impl_type_hpppoint_array.h"
#include "hozon/hmi/impl_type_hppobject_array.h"

namespace hozon {
namespace hmi {
struct HPP_Path_Struct {
    ::uint32_t pathID;
    ::hozon::hmi::HPPPoint_Array points;
    ::hozon::hmi::HPPObject_Array staticSRData;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(pathID);
        fun(points);
        fun(staticSRData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(pathID);
        fun(points);
        fun(staticSRData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("pathID", pathID);
        fun("points", points);
        fun("staticSRData", staticSRData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("pathID", pathID);
        fun("points", points);
        fun("staticSRData", staticSRData);
    }

    bool operator==(const ::hozon::hmi::HPP_Path_Struct& t) const
    {
        return (pathID == t.pathID) && (points == t.points) && (staticSRData == t.staticSRData);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HPP_PATH_STRUCT_H
