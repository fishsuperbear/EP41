/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHDECODEDSTRUCT_H
#define ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHDECODEDSTRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/camera/impl_type_pointdatastruct.h"
#include "ara/camera/impl_type_cameraheaderstruct.h"

namespace ara {
namespace camera {
struct CameraPublishDecodedStruct {
    ::UInt32 Height;
    ::UInt32 Width;
    ::UInt32 SendTimeHigh;
    ::UInt32 SendTimeLow;
    ::UInt32 FrameType;
    ::UInt32 DataSize;
    ::ara::camera::PointDataStruct RawData;
    ::ara::camera::CameraHeaderStruct CameraHeader;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Height);
        fun(Width);
        fun(SendTimeHigh);
        fun(SendTimeLow);
        fun(FrameType);
        fun(DataSize);
        fun(RawData);
        fun(CameraHeader);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Height);
        fun(Width);
        fun(SendTimeHigh);
        fun(SendTimeLow);
        fun(FrameType);
        fun(DataSize);
        fun(RawData);
        fun(CameraHeader);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Height", Height);
        fun("Width", Width);
        fun("SendTimeHigh", SendTimeHigh);
        fun("SendTimeLow", SendTimeLow);
        fun("FrameType", FrameType);
        fun("DataSize", DataSize);
        fun("RawData", RawData);
        fun("CameraHeader", CameraHeader);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Height", Height);
        fun("Width", Width);
        fun("SendTimeHigh", SendTimeHigh);
        fun("SendTimeLow", SendTimeLow);
        fun("FrameType", FrameType);
        fun("DataSize", DataSize);
        fun("RawData", RawData);
        fun("CameraHeader", CameraHeader);
    }

    bool operator==(const ::ara::camera::CameraPublishDecodedStruct& t) const
    {
        return (Height == t.Height) && (Width == t.Width) && (SendTimeHigh == t.SendTimeHigh) && (SendTimeLow == t.SendTimeLow) && (FrameType == t.FrameType) && (DataSize == t.DataSize) && (RawData == t.RawData) && (CameraHeader == t.CameraHeader);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHDECODEDSTRUCT_H
