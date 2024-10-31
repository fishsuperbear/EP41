/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_CAMERAHEADERSTRUCT_H
#define ARA_CAMERA_IMPL_TYPE_CAMERAHEADERSTRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/camera/impl_type_cameraheadtimestruct.h"
#include "impl_type_string.h"

namespace ara {
namespace camera {
struct CameraHeaderStruct {
    ::UInt32 Seq;
    ::ara::camera::CameraHeadTimeStruct Stamp;
    ::String FrameId;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Seq);
        fun(Stamp);
        fun(FrameId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Seq);
        fun(Stamp);
        fun(FrameId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Seq", Seq);
        fun("Stamp", Stamp);
        fun("FrameId", FrameId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Seq", Seq);
        fun("Stamp", Stamp);
        fun("FrameId", FrameId);
    }

    bool operator==(const ::ara::camera::CameraHeaderStruct& t) const
    {
        return (Seq == t.Seq) && (Stamp == t.Stamp) && (FrameId == t.FrameId);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_CAMERAHEADERSTRUCT_H
