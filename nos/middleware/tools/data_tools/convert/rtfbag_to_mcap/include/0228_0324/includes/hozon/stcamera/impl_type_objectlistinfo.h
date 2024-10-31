/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_OBJECTLISTINFO_H
#define HOZON_STCAMERA_IMPL_TYPE_OBJECTLISTINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace stcamera {
struct ObjectListInfo {
    ::Float cam_obj_list_delay;
    ::UInt32 cam_version_number;
    ::UInt8 cam_sensor_state;
    ::UInt8 cam_num_objects;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(cam_obj_list_delay);
        fun(cam_version_number);
        fun(cam_sensor_state);
        fun(cam_num_objects);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(cam_obj_list_delay);
        fun(cam_version_number);
        fun(cam_sensor_state);
        fun(cam_num_objects);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("cam_obj_list_delay", cam_obj_list_delay);
        fun("cam_version_number", cam_version_number);
        fun("cam_sensor_state", cam_sensor_state);
        fun("cam_num_objects", cam_num_objects);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("cam_obj_list_delay", cam_obj_list_delay);
        fun("cam_version_number", cam_version_number);
        fun("cam_sensor_state", cam_sensor_state);
        fun("cam_num_objects", cam_num_objects);
    }

    bool operator==(const ::hozon::stcamera::ObjectListInfo& t) const
    {
        return (fabs(static_cast<double>(cam_obj_list_delay - t.cam_obj_list_delay)) < DBL_EPSILON) && (cam_version_number == t.cam_version_number) && (cam_sensor_state == t.cam_sensor_state) && (cam_num_objects == t.cam_num_objects);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_OBJECTLISTINFO_H
