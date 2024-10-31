/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERA3D_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERA3D_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/object/impl_type_rect3d.h"
#include "hozon/composite/impl_type_vector3.h"
#include "hozon/composite/impl_type_floatarray9.h"
#include "hozon/object/impl_type_float32vector.h"
#include "hozon/object/impl_type_rect3dvector.h"
#include "hozon/object/impl_type_uint8vector.h"
#include "hozon/object/impl_type_point3f.h"
#include "hozon/common/impl_type_commontime.h"
#include "hozon/object/impl_type_feature.h"

namespace hozon {
namespace object {
struct ObjectCamera3D {
    ::UInt32 objectID;
    ::UInt8 type;
    ::UInt8 sub_type;
    ::UInt32 detCamSensor;
    ::Float confidence;
    ::hozon::object::Rect3D rect;
    ::hozon::composite::Vector3 velocity;
    ::hozon::composite::FloatArray9 velocity_unc;
    ::Float value;
    ::UInt8 movState;
    ::UInt8 brakeLightSt;
    ::UInt8 turnLightSt;
    ::UInt8 anchorPtInfo;
    ::hozon::object::Float32Vector associatedConf;
    ::hozon::object::Rect3DVector detectionKeyComBox;
    ::hozon::object::Uint8Vector detectionKeyComID;
    ::hozon::object::Point3f accel;
    ::hozon::composite::FloatArray9 accel_unc;
    ::hozon::common::CommonTime timeCreation;
    ::UInt8 isOnroad;
    ::Float onRoadProb;
    ::UInt8 isOccluded;
    ::Float occludedProb;
    ::UInt8 isTruncated;
    ::Float truncatedProb;
    ::UInt8 isSprinkler;
    ::Float sprinklerProb;
    ::Float illuminantDistance;
    ::Float existConfidence;
    ::UInt32 cipvSt;
    ::UInt32 cippSt;
    ::UInt32 age;
    ::hozon::object::Rect3DVector detObjectBox;
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
        fun(type);
        fun(sub_type);
        fun(detCamSensor);
        fun(confidence);
        fun(rect);
        fun(velocity);
        fun(velocity_unc);
        fun(value);
        fun(movState);
        fun(brakeLightSt);
        fun(turnLightSt);
        fun(anchorPtInfo);
        fun(associatedConf);
        fun(detectionKeyComBox);
        fun(detectionKeyComID);
        fun(accel);
        fun(accel_unc);
        fun(timeCreation);
        fun(isOnroad);
        fun(onRoadProb);
        fun(isOccluded);
        fun(occludedProb);
        fun(isTruncated);
        fun(truncatedProb);
        fun(isSprinkler);
        fun(sprinklerProb);
        fun(illuminantDistance);
        fun(existConfidence);
        fun(cipvSt);
        fun(cippSt);
        fun(age);
        fun(detObjectBox);
        fun(feature);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(type);
        fun(sub_type);
        fun(detCamSensor);
        fun(confidence);
        fun(rect);
        fun(velocity);
        fun(velocity_unc);
        fun(value);
        fun(movState);
        fun(brakeLightSt);
        fun(turnLightSt);
        fun(anchorPtInfo);
        fun(associatedConf);
        fun(detectionKeyComBox);
        fun(detectionKeyComID);
        fun(accel);
        fun(accel_unc);
        fun(timeCreation);
        fun(isOnroad);
        fun(onRoadProb);
        fun(isOccluded);
        fun(occludedProb);
        fun(isTruncated);
        fun(truncatedProb);
        fun(isSprinkler);
        fun(sprinklerProb);
        fun(illuminantDistance);
        fun(existConfidence);
        fun(cipvSt);
        fun(cippSt);
        fun(age);
        fun(detObjectBox);
        fun(feature);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("sub_type", sub_type);
        fun("detCamSensor", detCamSensor);
        fun("confidence", confidence);
        fun("rect", rect);
        fun("velocity", velocity);
        fun("velocity_unc", velocity_unc);
        fun("value", value);
        fun("movState", movState);
        fun("brakeLightSt", brakeLightSt);
        fun("turnLightSt", turnLightSt);
        fun("anchorPtInfo", anchorPtInfo);
        fun("associatedConf", associatedConf);
        fun("detectionKeyComBox", detectionKeyComBox);
        fun("detectionKeyComID", detectionKeyComID);
        fun("accel", accel);
        fun("accel_unc", accel_unc);
        fun("timeCreation", timeCreation);
        fun("isOnroad", isOnroad);
        fun("onRoadProb", onRoadProb);
        fun("isOccluded", isOccluded);
        fun("occludedProb", occludedProb);
        fun("isTruncated", isTruncated);
        fun("truncatedProb", truncatedProb);
        fun("isSprinkler", isSprinkler);
        fun("sprinklerProb", sprinklerProb);
        fun("illuminantDistance", illuminantDistance);
        fun("existConfidence", existConfidence);
        fun("cipvSt", cipvSt);
        fun("cippSt", cippSt);
        fun("age", age);
        fun("detObjectBox", detObjectBox);
        fun("feature", feature);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("sub_type", sub_type);
        fun("detCamSensor", detCamSensor);
        fun("confidence", confidence);
        fun("rect", rect);
        fun("velocity", velocity);
        fun("velocity_unc", velocity_unc);
        fun("value", value);
        fun("movState", movState);
        fun("brakeLightSt", brakeLightSt);
        fun("turnLightSt", turnLightSt);
        fun("anchorPtInfo", anchorPtInfo);
        fun("associatedConf", associatedConf);
        fun("detectionKeyComBox", detectionKeyComBox);
        fun("detectionKeyComID", detectionKeyComID);
        fun("accel", accel);
        fun("accel_unc", accel_unc);
        fun("timeCreation", timeCreation);
        fun("isOnroad", isOnroad);
        fun("onRoadProb", onRoadProb);
        fun("isOccluded", isOccluded);
        fun("occludedProb", occludedProb);
        fun("isTruncated", isTruncated);
        fun("truncatedProb", truncatedProb);
        fun("isSprinkler", isSprinkler);
        fun("sprinklerProb", sprinklerProb);
        fun("illuminantDistance", illuminantDistance);
        fun("existConfidence", existConfidence);
        fun("cipvSt", cipvSt);
        fun("cippSt", cippSt);
        fun("age", age);
        fun("detObjectBox", detObjectBox);
        fun("feature", feature);
    }

    bool operator==(const ::hozon::object::ObjectCamera3D& t) const
    {
        return (objectID == t.objectID) && (type == t.type) && (sub_type == t.sub_type) && (detCamSensor == t.detCamSensor) && (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (rect == t.rect) && (velocity == t.velocity) && (velocity_unc == t.velocity_unc) && (fabs(static_cast<double>(value - t.value)) < DBL_EPSILON) && (movState == t.movState) && (brakeLightSt == t.brakeLightSt) && (turnLightSt == t.turnLightSt) && (anchorPtInfo == t.anchorPtInfo) && (associatedConf == t.associatedConf) && (detectionKeyComBox == t.detectionKeyComBox) && (detectionKeyComID == t.detectionKeyComID) && (accel == t.accel) && (accel_unc == t.accel_unc) && (timeCreation == t.timeCreation) && (isOnroad == t.isOnroad) && (fabs(static_cast<double>(onRoadProb - t.onRoadProb)) < DBL_EPSILON) && (isOccluded == t.isOccluded) && (fabs(static_cast<double>(occludedProb - t.occludedProb)) < DBL_EPSILON) && (isTruncated == t.isTruncated) && (fabs(static_cast<double>(truncatedProb - t.truncatedProb)) < DBL_EPSILON) && (isSprinkler == t.isSprinkler) && (fabs(static_cast<double>(sprinklerProb - t.sprinklerProb)) < DBL_EPSILON) && (fabs(static_cast<double>(illuminantDistance - t.illuminantDistance)) < DBL_EPSILON) && (fabs(static_cast<double>(existConfidence - t.existConfidence)) < DBL_EPSILON) && (cipvSt == t.cipvSt) && (cippSt == t.cippSt) && (age == t.age) && (detObjectBox == t.detObjectBox) && (feature == t.feature);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTCAMERA3D_H
