/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERA2D_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERA2D_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "hozon/object/impl_type_rect2dvector.h"
#include "hozon/composite/impl_type_point2darray.h"
#include "impl_type_double.h"

namespace hozon {
namespace object {
struct ObjectCamera2D {
    ::UInt32 objectID;
    ::UInt8 type;
    ::UInt8 value;
    ::hozon::object::Rect2DVector rect;
    ::hozon::object::Rect2DVector detectionKeyComRect;
    ::hozon::composite::Point2DArray regressPoints;
    ::hozon::object::Rect2DVector regRect;
    ::Double confidence;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(objectID);
        fun(type);
        fun(value);
        fun(rect);
        fun(detectionKeyComRect);
        fun(regressPoints);
        fun(regRect);
        fun(confidence);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(type);
        fun(value);
        fun(rect);
        fun(detectionKeyComRect);
        fun(regressPoints);
        fun(regRect);
        fun(confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("value", value);
        fun("rect", rect);
        fun("detectionKeyComRect", detectionKeyComRect);
        fun("regressPoints", regressPoints);
        fun("regRect", regRect);
        fun("confidence", confidence);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("value", value);
        fun("rect", rect);
        fun("detectionKeyComRect", detectionKeyComRect);
        fun("regressPoints", regressPoints);
        fun("regRect", regRect);
        fun("confidence", confidence);
    }

    bool operator==(const ::hozon::object::ObjectCamera2D& t) const
    {
        return (objectID == t.objectID) && (type == t.type) && (value == t.value) && (rect == t.rect) && (detectionKeyComRect == t.detectionKeyComRect) && (regressPoints == t.regressPoints) && (regRect == t.regRect) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERA2D_H
