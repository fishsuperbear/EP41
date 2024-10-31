/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHIMAGEDATASTRUCT_H
#define ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHIMAGEDATASTRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_rawbuffer.h"
#include "ara/camera/impl_type_uint32vector.h"

namespace ara {
namespace camera {
struct CameraPublishImageDataStruct {
    ::UInt32 version;
    ::UInt32 seq;
    ::rawBuffer *head;
    ::ara::camera::Uint32Vector type;

    static bool IsPlane()
    {
        return false;
    }

    ::rawBuffer* GetMbufPtr() const
    {
        return head;
    }

    void SetMbufPtr(::rawBuffer *p)
    {
        head = p;
    }

    using IsDpRawDataTag = void;
    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(version);
        fun(seq);
        fun(type);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(version);
        fun(seq);
        fun(type);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("version", version);
        fun("seq", seq);
        fun("type", type);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("version", version);
        fun("seq", seq);
        fun("type", type);
    }

    bool operator==(const ::ara::camera::CameraPublishImageDataStruct& t) const
    {
        return (version == t.version) && (seq == t.seq) && (head == t.head) && (type == t.type);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_CAMERAPUBLISHIMAGEDATASTRUCT_H
