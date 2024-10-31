/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHENCODEDSTRUCT_H
#define ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHENCODEDSTRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_string.h"
#include "ara/camera/impl_type_pointdatastruct.h"
#include "ara/camera/impl_type_cameraheaderstruct.h"

namespace ara {
namespace camera {
struct CameraPublishEncodedStruct {
    ::UInt32 DataSize;
    ::UInt32 SendTimeHigh;
    ::UInt32 SendTimeLow;
    ::UInt32 FrameType;
    ::String VideoFormat;
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
        fun(DataSize);
        fun(SendTimeHigh);
        fun(SendTimeLow);
        fun(FrameType);
        fun(VideoFormat);
        fun(RawData);
        fun(CameraHeader);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(DataSize);
        fun(SendTimeHigh);
        fun(SendTimeLow);
        fun(FrameType);
        fun(VideoFormat);
        fun(RawData);
        fun(CameraHeader);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("DataSize", DataSize);
        fun("SendTimeHigh", SendTimeHigh);
        fun("SendTimeLow", SendTimeLow);
        fun("FrameType", FrameType);
        fun("VideoFormat", VideoFormat);
        fun("RawData", RawData);
        fun("CameraHeader", CameraHeader);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("DataSize", DataSize);
        fun("SendTimeHigh", SendTimeHigh);
        fun("SendTimeLow", SendTimeLow);
        fun("FrameType", FrameType);
        fun("VideoFormat", VideoFormat);
        fun("RawData", RawData);
        fun("CameraHeader", CameraHeader);
    }

    bool operator==(const ::ara::camera::CameraPublishEncodedStruct& t) const
    {
        return (DataSize == t.DataSize) && (SendTimeHigh == t.SendTimeHigh) && (SendTimeLow == t.SendTimeLow) && (FrameType == t.FrameType) && (VideoFormat == t.VideoFormat) && (RawData == t.RawData) && (CameraHeader == t.CameraHeader);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHENCODEDSTRUCT_H
