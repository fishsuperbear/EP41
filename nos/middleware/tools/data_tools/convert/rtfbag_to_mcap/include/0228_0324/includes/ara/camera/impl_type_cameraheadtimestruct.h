/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_CAMERAHEADTIMESTRUCT_H
#define ARA_CAMERA_IMPL_TYPE_CAMERAHEADTIMESTRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"

namespace ara {
namespace camera {
struct CameraHeadTimeStruct {
    ::UInt32 Sec;
    ::UInt32 Nsec;
    ::UInt32 ExpStartS;
    ::UInt32 ExpStartNs;
    ::UInt32 ExpEndS;
    ::UInt32 ExpEndNs;
    ::UInt32 Shutter1;
    ::UInt32 Shutter2;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Sec);
        fun(Nsec);
        fun(ExpStartS);
        fun(ExpStartNs);
        fun(ExpEndS);
        fun(ExpEndNs);
        fun(Shutter1);
        fun(Shutter2);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Sec);
        fun(Nsec);
        fun(ExpStartS);
        fun(ExpStartNs);
        fun(ExpEndS);
        fun(ExpEndNs);
        fun(Shutter1);
        fun(Shutter2);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Sec", Sec);
        fun("Nsec", Nsec);
        fun("ExpStartS", ExpStartS);
        fun("ExpStartNs", ExpStartNs);
        fun("ExpEndS", ExpEndS);
        fun("ExpEndNs", ExpEndNs);
        fun("Shutter1", Shutter1);
        fun("Shutter2", Shutter2);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Sec", Sec);
        fun("Nsec", Nsec);
        fun("ExpStartS", ExpStartS);
        fun("ExpStartNs", ExpStartNs);
        fun("ExpEndS", ExpEndS);
        fun("ExpEndNs", ExpEndNs);
        fun("Shutter1", Shutter1);
        fun("Shutter2", Shutter2);
    }

    bool operator==(const ::ara::camera::CameraHeadTimeStruct& t) const
    {
        return (Sec == t.Sec) && (Nsec == t.Nsec) && (ExpStartS == t.ExpStartS) && (ExpStartNs == t.ExpStartNs) && (ExpEndS == t.ExpEndS) && (ExpEndNs == t.ExpEndNs) && (Shutter1 == t.Shutter1) && (Shutter2 == t.Shutter2);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_CAMERAHEADTIMESTRUCT_H
