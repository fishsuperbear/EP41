/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTLIDAR3D_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTLIDAR3D_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/object/impl_type_rect3d.h"
#include "hozon/common/impl_type_commontime.h"
#include "hozon/object/impl_type_pointvector.h"
#include "hozon/composite/impl_type_vector3.h"
#include "hozon/composite/impl_type_floatarray9.h"
#include "impl_type_boolean.h"
#include "hozon/object/impl_type_feature.h"

namespace hozon {
namespace object {
struct ObjectLidar3D {
    ::UInt32 objectID;
    ::UInt8 cls;
    ::UInt8 types;
    ::Float confidence;
    ::UInt8 coordinate;
    ::Float existenceProbability;
    ::hozon::object::Rect3D rect;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::object::pointVector contourPoints;
    ::UInt8 movingState;
    ::UInt8 trackState;
    ::hozon::composite::Vector3 velocity;
    ::hozon::composite::FloatArray9 velocity_unc;
    ::hozon::composite::Vector3 accel;
    ::hozon::composite::FloatArray9 accel_unc;
    ::Boolean isBackground;
    ::Boolean isOccluded;
    ::Float occludedProb;
    ::Boolean isTruncated;
    ::Float truncatedProb;
    ::hozon::object::Feature feature;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(objectID);
        fun(cls);
        fun(types);
        fun(confidence);
        fun(coordinate);
        fun(existenceProbability);
        fun(rect);
        fun(timeCreation);
        fun(contourPoints);
        fun(movingState);
        fun(trackState);
        fun(velocity);
        fun(velocity_unc);
        fun(accel);
        fun(accel_unc);
        fun(isBackground);
        fun(isOccluded);
        fun(occludedProb);
        fun(isTruncated);
        fun(truncatedProb);
        fun(feature);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(cls);
        fun(types);
        fun(confidence);
        fun(coordinate);
        fun(existenceProbability);
        fun(rect);
        fun(timeCreation);
        fun(contourPoints);
        fun(movingState);
        fun(trackState);
        fun(velocity);
        fun(velocity_unc);
        fun(accel);
        fun(accel_unc);
        fun(isBackground);
        fun(isOccluded);
        fun(occludedProb);
        fun(isTruncated);
        fun(truncatedProb);
        fun(feature);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("cls", cls);
        fun("types", types);
        fun("confidence", confidence);
        fun("coordinate", coordinate);
        fun("existenceProbability", existenceProbability);
        fun("rect", rect);
        fun("timeCreation", timeCreation);
        fun("contourPoints", contourPoints);
        fun("movingState", movingState);
        fun("trackState", trackState);
        fun("velocity", velocity);
        fun("velocity_unc", velocity_unc);
        fun("accel", accel);
        fun("accel_unc", accel_unc);
        fun("isBackground", isBackground);
        fun("isOccluded", isOccluded);
        fun("occludedProb", occludedProb);
        fun("isTruncated", isTruncated);
        fun("truncatedProb", truncatedProb);
        fun("feature", feature);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("cls", cls);
        fun("types", types);
        fun("confidence", confidence);
        fun("coordinate", coordinate);
        fun("existenceProbability", existenceProbability);
        fun("rect", rect);
        fun("timeCreation", timeCreation);
        fun("contourPoints", contourPoints);
        fun("movingState", movingState);
        fun("trackState", trackState);
        fun("velocity", velocity);
        fun("velocity_unc", velocity_unc);
        fun("accel", accel);
        fun("accel_unc", accel_unc);
        fun("isBackground", isBackground);
        fun("isOccluded", isOccluded);
        fun("occludedProb", occludedProb);
        fun("isTruncated", isTruncated);
        fun("truncatedProb", truncatedProb);
        fun("feature", feature);
    }

    bool operator==(const ::hozon::object::ObjectLidar3D& t) const
    {
        return (objectID == t.objectID) && (cls == t.cls) && (types == t.types) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (coordinate == t.coordinate) && (fabs(static_cast<double>(existenceProbability - t.existenceProbability)) < DBL_EPSILON) && (rect == t.rect) && (timeCreation == t.timeCreation) && (contourPoints == t.contourPoints) && (movingState == t.movingState) && (trackState == t.trackState) && (velocity == t.velocity) && (velocity_unc == t.velocity_unc) && (accel == t.accel) && (accel_unc == t.accel_unc) && (isBackground == t.isBackground) && (isOccluded == t.isOccluded) && (fabs(static_cast<double>(occludedProb - t.occludedProb)) < DBL_EPSILON) && (isTruncated == t.isTruncated) && (fabs(static_cast<double>(truncatedProb - t.truncatedProb)) < DBL_EPSILON) && (feature == t.feature);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTLIDAR3D_H
