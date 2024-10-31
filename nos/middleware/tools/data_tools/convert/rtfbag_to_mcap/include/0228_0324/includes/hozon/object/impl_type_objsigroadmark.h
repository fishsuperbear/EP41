/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJSIGROADMARK_H
#define HOZON_OBJECT_IMPL_TYPE_OBJSIGROADMARK_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "hozon/object/impl_type_polygonvector.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point2darrayarray.h"

namespace hozon {
namespace object {
struct ObjSigRoadMark {
    ::UInt32 objectID;
    ::Double long_dist;
    ::Double lat_dist;
    ::hozon::object::polygonVector polygon;
    ::UInt8 type;
    ::Double confidence;
    ::hozon::composite::Point2DArrayArray imagerect;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(objectID);
        fun(long_dist);
        fun(lat_dist);
        fun(polygon);
        fun(type);
        fun(confidence);
        fun(imagerect);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(long_dist);
        fun(lat_dist);
        fun(polygon);
        fun(type);
        fun(confidence);
        fun(imagerect);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("long_dist", long_dist);
        fun("lat_dist", lat_dist);
        fun("polygon", polygon);
        fun("type", type);
        fun("confidence", confidence);
        fun("imagerect", imagerect);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("long_dist", long_dist);
        fun("lat_dist", lat_dist);
        fun("polygon", polygon);
        fun("type", type);
        fun("confidence", confidence);
        fun("imagerect", imagerect);
    }

    bool operator==(const ::hozon::object::ObjSigRoadMark& t) const
    {
        return (objectID == t.objectID) && (fabs(static_cast<double>(long_dist - t.long_dist)) < DBL_EPSILON) && (fabs(static_cast<double>(lat_dist - t.lat_dist)) < DBL_EPSILON) && (polygon == t.polygon) && (type == t.type) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (imagerect == t.imagerect);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJSIGROADMARK_H
