/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RADAR_IMPL_TYPE_RADARSTATE_H
#define ARA_RADAR_IMPL_TYPE_RADARSTATE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"

namespace ara {
namespace radar {
struct RadarState {
    ::UInt8 sensorId;
    ::UInt8 nvmReadStatus;
    ::UInt8 nvmWriteStatus;
    ::UInt8 persistentError;
    ::UInt8 temporaryError;
    ::UInt8 interferenceError;
    ::UInt8 temperatureError;
    ::UInt8 voltageError;
    ::UInt16 maxDistance;
    ::UInt8 sortIndex;
    ::UInt8 radarPower;
    ::UInt8 ctrlRelay;
    ::UInt8 outputType;
    ::UInt8 sendQuality;
    ::UInt8 sendExtinfo;
    ::UInt8 motionRxState;
    ::UInt8 rcsThreshold;
    ::UInt8 blockError;
    ::UInt8 broadcastError;
    ::UInt8 electricAxisError;
    ::UInt8 configError;
    ::UInt8 calibrationError;
    ::UInt8 connectorDirection;
    ::UInt8 can0WorkMode;
    ::UInt8 can1WorkMode;
    ::UInt8 dualCanMode;
    ::UInt8 timmingMode;
    ::UInt8 powerMode;
    ::UInt8 performanceMode;
    ::UInt8 radarPosition;
    ::UInt8 hwError;
    ::UInt8 modulationStatus;
    ::UInt8 failureFlag;
    ::UInt8 sleepSig;
    ::Float objAziAngleCalib;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(sensorId);
        fun(nvmReadStatus);
        fun(nvmWriteStatus);
        fun(persistentError);
        fun(temporaryError);
        fun(interferenceError);
        fun(temperatureError);
        fun(voltageError);
        fun(maxDistance);
        fun(sortIndex);
        fun(radarPower);
        fun(ctrlRelay);
        fun(outputType);
        fun(sendQuality);
        fun(sendExtinfo);
        fun(motionRxState);
        fun(rcsThreshold);
        fun(blockError);
        fun(broadcastError);
        fun(electricAxisError);
        fun(configError);
        fun(calibrationError);
        fun(connectorDirection);
        fun(can0WorkMode);
        fun(can1WorkMode);
        fun(dualCanMode);
        fun(timmingMode);
        fun(powerMode);
        fun(performanceMode);
        fun(radarPosition);
        fun(hwError);
        fun(modulationStatus);
        fun(failureFlag);
        fun(sleepSig);
        fun(objAziAngleCalib);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(sensorId);
        fun(nvmReadStatus);
        fun(nvmWriteStatus);
        fun(persistentError);
        fun(temporaryError);
        fun(interferenceError);
        fun(temperatureError);
        fun(voltageError);
        fun(maxDistance);
        fun(sortIndex);
        fun(radarPower);
        fun(ctrlRelay);
        fun(outputType);
        fun(sendQuality);
        fun(sendExtinfo);
        fun(motionRxState);
        fun(rcsThreshold);
        fun(blockError);
        fun(broadcastError);
        fun(electricAxisError);
        fun(configError);
        fun(calibrationError);
        fun(connectorDirection);
        fun(can0WorkMode);
        fun(can1WorkMode);
        fun(dualCanMode);
        fun(timmingMode);
        fun(powerMode);
        fun(performanceMode);
        fun(radarPosition);
        fun(hwError);
        fun(modulationStatus);
        fun(failureFlag);
        fun(sleepSig);
        fun(objAziAngleCalib);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("sensorId", sensorId);
        fun("nvmReadStatus", nvmReadStatus);
        fun("nvmWriteStatus", nvmWriteStatus);
        fun("persistentError", persistentError);
        fun("temporaryError", temporaryError);
        fun("interferenceError", interferenceError);
        fun("temperatureError", temperatureError);
        fun("voltageError", voltageError);
        fun("maxDistance", maxDistance);
        fun("sortIndex", sortIndex);
        fun("radarPower", radarPower);
        fun("ctrlRelay", ctrlRelay);
        fun("outputType", outputType);
        fun("sendQuality", sendQuality);
        fun("sendExtinfo", sendExtinfo);
        fun("motionRxState", motionRxState);
        fun("rcsThreshold", rcsThreshold);
        fun("blockError", blockError);
        fun("broadcastError", broadcastError);
        fun("electricAxisError", electricAxisError);
        fun("configError", configError);
        fun("calibrationError", calibrationError);
        fun("connectorDirection", connectorDirection);
        fun("can0WorkMode", can0WorkMode);
        fun("can1WorkMode", can1WorkMode);
        fun("dualCanMode", dualCanMode);
        fun("timmingMode", timmingMode);
        fun("powerMode", powerMode);
        fun("performanceMode", performanceMode);
        fun("radarPosition", radarPosition);
        fun("hwError", hwError);
        fun("modulationStatus", modulationStatus);
        fun("failureFlag", failureFlag);
        fun("sleepSig", sleepSig);
        fun("objAziAngleCalib", objAziAngleCalib);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("sensorId", sensorId);
        fun("nvmReadStatus", nvmReadStatus);
        fun("nvmWriteStatus", nvmWriteStatus);
        fun("persistentError", persistentError);
        fun("temporaryError", temporaryError);
        fun("interferenceError", interferenceError);
        fun("temperatureError", temperatureError);
        fun("voltageError", voltageError);
        fun("maxDistance", maxDistance);
        fun("sortIndex", sortIndex);
        fun("radarPower", radarPower);
        fun("ctrlRelay", ctrlRelay);
        fun("outputType", outputType);
        fun("sendQuality", sendQuality);
        fun("sendExtinfo", sendExtinfo);
        fun("motionRxState", motionRxState);
        fun("rcsThreshold", rcsThreshold);
        fun("blockError", blockError);
        fun("broadcastError", broadcastError);
        fun("electricAxisError", electricAxisError);
        fun("configError", configError);
        fun("calibrationError", calibrationError);
        fun("connectorDirection", connectorDirection);
        fun("can0WorkMode", can0WorkMode);
        fun("can1WorkMode", can1WorkMode);
        fun("dualCanMode", dualCanMode);
        fun("timmingMode", timmingMode);
        fun("powerMode", powerMode);
        fun("performanceMode", performanceMode);
        fun("radarPosition", radarPosition);
        fun("hwError", hwError);
        fun("modulationStatus", modulationStatus);
        fun("failureFlag", failureFlag);
        fun("sleepSig", sleepSig);
        fun("objAziAngleCalib", objAziAngleCalib);
    }

    bool operator==(const ::ara::radar::RadarState& t) const
    {
        return (sensorId == t.sensorId) && (nvmReadStatus == t.nvmReadStatus) && (nvmWriteStatus == t.nvmWriteStatus) && (persistentError == t.persistentError) && (temporaryError == t.temporaryError) && (interferenceError == t.interferenceError) && (temperatureError == t.temperatureError) && (voltageError == t.voltageError) && (maxDistance == t.maxDistance) && (sortIndex == t.sortIndex) && (radarPower == t.radarPower) && (ctrlRelay == t.ctrlRelay) && (outputType == t.outputType) && (sendQuality == t.sendQuality) && (sendExtinfo == t.sendExtinfo) && (motionRxState == t.motionRxState) && (rcsThreshold == t.rcsThreshold) && (blockError == t.blockError) && (broadcastError == t.broadcastError) && (electricAxisError == t.electricAxisError) && (configError == t.configError) && (calibrationError == t.calibrationError) && (connectorDirection == t.connectorDirection) && (can0WorkMode == t.can0WorkMode) && (can1WorkMode == t.can1WorkMode) && (dualCanMode == t.dualCanMode) && (timmingMode == t.timmingMode) && (powerMode == t.powerMode) && (performanceMode == t.performanceMode) && (radarPosition == t.radarPosition) && (hwError == t.hwError) && (modulationStatus == t.modulationStatus) && (failureFlag == t.failureFlag) && (sleepSig == t.sleepSig) && (fabs(static_cast<double>(objAziAngleCalib - t.objAziAngleCalib)) < DBL_EPSILON);
    }
};
} // namespace radar
} // namespace ara


#endif // ARA_RADAR_IMPL_TYPE_RADARSTATE_H
