/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_STCAMERADATA_H
#define HOZON_STCAMERA_IMPL_TYPE_STCAMERADATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64.h"
#include "hozon/stcamera/impl_type_abevent.h"
#include "hozon/stcamera/impl_type_abdinfo.h"
#include "hozon/stcamera/impl_type_boundaryarray.h"
#include "hozon/stcamera/impl_type_camobjectarray.h"
#include "hozon/stcamera/impl_type_objectlistinfo.h"
#include "hozon/stcamera/impl_type_ved.h"

namespace hozon {
namespace stcamera {
struct StCameraData {
    ::UInt64 time;
    ::hozon::stcamera::ABEvent ab_event;
    ::hozon::stcamera::ABDInfo abd_info;
    ::hozon::stcamera::BoundaryArray boundaries;
    ::hozon::stcamera::CamObjectArray cam_objects;
    ::hozon::stcamera::ObjectListInfo object_list_info;
    ::hozon::stcamera::VED ved;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time);
        fun(ab_event);
        fun(abd_info);
        fun(boundaries);
        fun(cam_objects);
        fun(object_list_info);
        fun(ved);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time);
        fun(ab_event);
        fun(abd_info);
        fun(boundaries);
        fun(cam_objects);
        fun(object_list_info);
        fun(ved);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time", time);
        fun("ab_event", ab_event);
        fun("abd_info", abd_info);
        fun("boundaries", boundaries);
        fun("cam_objects", cam_objects);
        fun("object_list_info", object_list_info);
        fun("ved", ved);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time", time);
        fun("ab_event", ab_event);
        fun("abd_info", abd_info);
        fun("boundaries", boundaries);
        fun("cam_objects", cam_objects);
        fun("object_list_info", object_list_info);
        fun("ved", ved);
    }

    bool operator==(const ::hozon::stcamera::StCameraData& t) const
    {
        return (time == t.time) && (ab_event == t.ab_event) && (abd_info == t.abd_info) && (boundaries == t.boundaries) && (cam_objects == t.cam_objects) && (object_list_info == t.object_list_info) && (ved == t.ved);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_STCAMERADATA_H
