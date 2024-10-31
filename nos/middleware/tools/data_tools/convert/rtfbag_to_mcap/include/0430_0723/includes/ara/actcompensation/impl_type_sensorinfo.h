/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_SENSORINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_SENSORINFO_H
#include <cfloat>
#include <cmath>
#include "ara/actcompensation/impl_type_float32withvalid.h"
#include "ara/actcompensation/impl_type_uint16withvalid.h"
#include "ara/actcompensation/impl_type_uint8withvalid.h"

namespace ara {
namespace actcompensation {
struct SensorInfo {
    ::ara::actcompensation::Float32WithValid insideTemperature;
    ::ara::actcompensation::Float32WithValid outsideTemperature;
    ::ara::actcompensation::Uint16WithValid relativeHumidity;
    ::ara::actcompensation::Uint8WithValid rainLightSensor;
    ::ara::actcompensation::Uint8WithValid rainfall;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(insideTemperature);
        fun(outsideTemperature);
        fun(relativeHumidity);
        fun(rainLightSensor);
        fun(rainfall);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(insideTemperature);
        fun(outsideTemperature);
        fun(relativeHumidity);
        fun(rainLightSensor);
        fun(rainfall);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("insideTemperature", insideTemperature);
        fun("outsideTemperature", outsideTemperature);
        fun("relativeHumidity", relativeHumidity);
        fun("rainLightSensor", rainLightSensor);
        fun("rainfall", rainfall);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("insideTemperature", insideTemperature);
        fun("outsideTemperature", outsideTemperature);
        fun("relativeHumidity", relativeHumidity);
        fun("rainLightSensor", rainLightSensor);
        fun("rainfall", rainfall);
    }

    bool operator==(const ::ara::actcompensation::SensorInfo& t) const
    {
        return (insideTemperature == t.insideTemperature) && (outsideTemperature == t.outsideTemperature) && (relativeHumidity == t.relativeHumidity) && (rainLightSensor == t.rainLightSensor) && (rainfall == t.rainfall);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_SENSORINFO_H
