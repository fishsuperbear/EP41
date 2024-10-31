/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DISPLAY_IMPL_TYPE_DISPLAYIMAGEMBUFSTRUCT_H
#define ARA_DISPLAY_IMPL_TYPE_DISPLAYIMAGEMBUFSTRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_uint64_t.h"
#include "impl_type_rawbuffer.h"
#include "impl_type_uint8_t.h"
#include "ara/display/impl_type_imageformattypee.h"
#include "ara/display/impl_type_rotatemodee.h"

namespace ara {
namespace display {
struct DisplayImageMbufStruct {
    ::uint32_t Height;
    ::uint32_t Width;
    ::uint64_t SendTimeStamp;
    ::uint32_t DataSize;
    ::rawBuffer *RawData;
    ::uint32_t Seq;
    ::uint8_t ChannelId;
    ::ara::display::ImageFormatTypeE InputFormat;
    ::ara::display::ImageFormatTypeE OutputFormat;
    ::ara::display::RotateModeE RotateMode;

    static bool IsPlane()
    {
        return true;
    }

    ::rawBuffer* GetMbufPtr() const
    {
        return RawData;
    }

    void SetMbufPtr(::rawBuffer *p)
    {
        RawData = p;
    }

    using IsDpRawDataTag = void;
    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Height);
        fun(Width);
        fun(SendTimeStamp);
        fun(DataSize);
        fun(Seq);
        fun(ChannelId);
        fun(InputFormat);
        fun(OutputFormat);
        fun(RotateMode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Height);
        fun(Width);
        fun(SendTimeStamp);
        fun(DataSize);
        fun(Seq);
        fun(ChannelId);
        fun(InputFormat);
        fun(OutputFormat);
        fun(RotateMode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Height", Height);
        fun("Width", Width);
        fun("SendTimeStamp", SendTimeStamp);
        fun("DataSize", DataSize);
        fun("Seq", Seq);
        fun("ChannelId", ChannelId);
        fun("InputFormat", InputFormat);
        fun("OutputFormat", OutputFormat);
        fun("RotateMode", RotateMode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Height", Height);
        fun("Width", Width);
        fun("SendTimeStamp", SendTimeStamp);
        fun("DataSize", DataSize);
        fun("Seq", Seq);
        fun("ChannelId", ChannelId);
        fun("InputFormat", InputFormat);
        fun("OutputFormat", OutputFormat);
        fun("RotateMode", RotateMode);
    }

    bool operator==(const ::ara::display::DisplayImageMbufStruct& t) const
    {
        return (Height == t.Height) && (Width == t.Width) && (SendTimeStamp == t.SendTimeStamp) && (DataSize == t.DataSize) && (RawData == t.RawData) && (Seq == t.Seq) && (ChannelId == t.ChannelId) && (InputFormat == t.InputFormat) && (OutputFormat == t.OutputFormat) && (RotateMode == t.RotateMode);
    }
};
} // namespace display
} // namespace ara


#endif // ARA_DISPLAY_IMPL_TYPE_DISPLAYIMAGEMBUFSTRUCT_H
