/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_OBJECTFUSION_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_OBJECTFUSION_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "hozon/soc_mcu/impl_type_rect3d_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_point2f_soc_mcu.h"
#include "hozon/common/impl_type_commontime.h"

namespace hozon {
namespace soc_mcu {
struct ObjectFusion_soc_mcu {
    ::uint8_t objectID;
    ::UInt8 type;
    ::UInt32 detectSensor_cur;
    ::UInt32 detectSensor_his;
    ::UInt8 mainten_status;
    ::uint8_t type_confidence;
    ::uint8_t existenceProbability;
    ::hozon::soc_mcu::Rect3D_soc_mcu rect;
    ::hozon::soc_mcu::Point2f_soc_mcu velocity;
    ::hozon::soc_mcu::Point2f_soc_mcu accel;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::common::CommonTime lastUpdated;
    ::UInt8 mottionPattern;
    ::UInt8 MotionPatternHistory;
    ::UInt8 BrakeLightSt;
    ::UInt8 TurnLightSt;
    ::UInt8 nearside;
    ::UInt32 Age;

    static bool IsPlane()
    {
        return true;
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
        fun(mottionPattern);
        fun(MotionPatternHistory);
        fun(BrakeLightSt);
        fun(TurnLightSt);
        fun(nearside);
        fun(Age);
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
        fun(mottionPattern);
        fun(MotionPatternHistory);
        fun(BrakeLightSt);
        fun(TurnLightSt);
        fun(nearside);
        fun(Age);
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
        fun("mottionPattern", mottionPattern);
        fun("MotionPatternHistory", MotionPatternHistory);
        fun("BrakeLightSt", BrakeLightSt);
        fun("TurnLightSt", TurnLightSt);
        fun("nearside", nearside);
        fun("Age", Age);
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
        fun("mottionPattern", mottionPattern);
        fun("MotionPatternHistory", MotionPatternHistory);
        fun("BrakeLightSt", BrakeLightSt);
        fun("TurnLightSt", TurnLightSt);
        fun("nearside", nearside);
        fun("Age", Age);
    }

    bool operator==(const ::hozon::soc_mcu::ObjectFusion_soc_mcu& t) const
    {
        return (objectID == t.objectID) && (type == t.type) && (detectSensor_cur == t.detectSensor_cur) && (detectSensor_his == t.detectSensor_his) && (mainten_status == t.mainten_status) && (type_confidence == t.type_confidence) && (existenceProbability == t.existenceProbability) && (rect == t.rect) && (velocity == t.velocity) && (accel == t.accel) && (timeCreation == t.timeCreation) && (lastUpdated == t.lastUpdated) && (mottionPattern == t.mottionPattern) && (MotionPatternHistory == t.MotionPatternHistory) && (BrakeLightSt == t.BrakeLightSt) && (TurnLightSt == t.TurnLightSt) && (nearside == t.nearside) && (Age == t.Age);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_OBJECTFUSION_SOC_MCU_H
