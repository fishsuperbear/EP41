/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTFUSION_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTFUSION_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/object/impl_type_rect3d.h"
#include "hozon/composite/impl_type_vector3.h"
#include "hozon/common/impl_type_commontime.h"
#include "hozon/object/impl_type_sensoridvector.h"
#include "hozon/object/impl_type_float32vector.h"

namespace hozon {
namespace object {
struct ObjectFusion {
    ::UInt32 objectID;
    ::UInt8 type;
    ::UInt32 detectSensor_cur;
    ::UInt32 detectSensor_his;
    ::UInt8 mainten_status;
    ::Float type_confidence;
    ::Float existenceProbability;
    ::hozon::object::Rect3D rect;
    ::hozon::composite::Vector3 velocity;
    ::hozon::composite::Vector3 accel;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::common::CommonTime lastUpdated;
    ::hozon::object::sensorIDVector sensorIDs;
    ::UInt8 mottionPattern;
    ::UInt8 MotionPatternHistory;
    ::UInt8 BrakeLightSt;
    ::UInt8 TurnLightSt;
    ::UInt8 nearside;
    ::hozon::object::Float32Vector associatedConf;
    ::UInt32 age;

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
        fun(detectSensor_cur);
        fun(detectSensor_his);
        fun(mainten_status);
        fun(type_confidence);
        fun(existenceProbability);
        fun(rect);
        fun(velocity);
        fun(accel);
        fun(timeCreation);
        fun(lastUpdated);
        fun(sensorIDs);
        fun(mottionPattern);
        fun(MotionPatternHistory);
        fun(BrakeLightSt);
        fun(TurnLightSt);
        fun(nearside);
        fun(associatedConf);
        fun(age);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(objectID);
        fun(type);
        fun(detectSensor_cur);
        fun(detectSensor_his);
        fun(mainten_status);
        fun(type_confidence);
        fun(existenceProbability);
        fun(rect);
        fun(velocity);
        fun(accel);
        fun(timeCreation);
        fun(lastUpdated);
        fun(sensorIDs);
        fun(mottionPattern);
        fun(MotionPatternHistory);
        fun(BrakeLightSt);
        fun(TurnLightSt);
        fun(nearside);
        fun(associatedConf);
        fun(age);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("detectSensor_cur", detectSensor_cur);
        fun("detectSensor_his", detectSensor_his);
        fun("mainten_status", mainten_status);
        fun("type_confidence", type_confidence);
        fun("existenceProbability", existenceProbability);
        fun("rect", rect);
        fun("velocity", velocity);
        fun("accel", accel);
        fun("timeCreation", timeCreation);
        fun("lastUpdated", lastUpdated);
        fun("sensorIDs", sensorIDs);
        fun("mottionPattern", mottionPattern);
        fun("MotionPatternHistory", MotionPatternHistory);
        fun("BrakeLightSt", BrakeLightSt);
        fun("TurnLightSt", TurnLightSt);
        fun("nearside", nearside);
        fun("associatedConf", associatedConf);
        fun("age", age);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("objectID", objectID);
        fun("type", type);
        fun("detectSensor_cur", detectSensor_cur);
        fun("detectSensor_his", detectSensor_his);
        fun("mainten_status", mainten_status);
        fun("type_confidence", type_confidence);
        fun("existenceProbability", existenceProbability);
        fun("rect", rect);
        fun("velocity", velocity);
        fun("accel", accel);
        fun("timeCreation", timeCreation);
        fun("lastUpdated", lastUpdated);
        fun("sensorIDs", sensorIDs);
        fun("mottionPattern", mottionPattern);
        fun("MotionPatternHistory", MotionPatternHistory);
        fun("BrakeLightSt", BrakeLightSt);
        fun("TurnLightSt", TurnLightSt);
        fun("nearside", nearside);
        fun("associatedConf", associatedConf);
        fun("age", age);
    }

    bool operator==(const ::hozon::object::ObjectFusion& t) const
    {
        return (objectID == t.objectID) && (type == t.type) && (detectSensor_cur == t.detectSensor_cur) && (detectSensor_his == t.detectSensor_his) && (mainten_status == t.mainten_status) && (fabs(static_cast<double>(type_confidence - t.type_confidence)) < DBL_EPSILON) && (fabs(static_cast<double>(existenceProbability - t.existenceProbability)) < DBL_EPSILON) && (rect == t.rect) && (velocity == t.velocity) && (accel == t.accel) && (timeCreation == t.timeCreation) && (lastUpdated == t.lastUpdated) && (sensorIDs == t.sensorIDs) && (mottionPattern == t.mottionPattern) && (MotionPatternHistory == t.MotionPatternHistory) && (BrakeLightSt == t.BrakeLightSt) && (TurnLightSt == t.TurnLightSt) && (nearside == t.nearside) && (associatedConf == t.associatedConf) && (age == t.age);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTFUSION_H
