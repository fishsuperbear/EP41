/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_BRAKEINFO_H
#define ARA_CHASSIS_IMPL_TYPE_BRAKEINFO_H
#include <cfloat>
#include <cmath>
#include "ara/chassis/impl_type_float32withvalid.h"
#include "ara/chassis/impl_type_uint8withvalid.h"
#include "impl_type_uint8.h"
#include "ara/chassis/impl_type_wheelspeed.h"
#include "impl_type_float.h"
#include "impl_type_int32.h"
#include "impl_type_boolean.h"

namespace ara {
namespace chassis {
struct BrakeInfo {
    ::ara::chassis::Float32WithValid brakePedal;
    ::ara::chassis::Uint8WithValid brakePedalSwitch;
    ::ara::chassis::Uint8WithValid driverOverride;
    ::ara::chassis::Float32WithValid mastCylinderPressure;
    ::ara::chassis::Float32WithValid mastCylinderPosition;
    ::ara::chassis::Float32WithValid brakeTorque;
    ::ara::chassis::Float32WithValid brakeForce;
    ::ara::chassis::Float32WithValid brakeTorqueMax;
    ::ara::chassis::Float32WithValid driverDesiredTorque;
    ::ara::chassis::Float32WithValid brakeTotalTqReq;
    ::ara::chassis::Float32WithValid elecBrkTqReq;
    ::ara::chassis::Float32WithValid hydrauBrkTqReq;
    ::UInt8 brakeOverHeatStatus;
    ::UInt8 aebAvl;
    ::UInt8 aebActv;
    ::UInt8 prefillAvl;
    ::UInt8 prefillActv;
    ::UInt8 brkCtrlAvl;
    ::UInt8 brkCtrlActv;
    ::UInt8 espAbortFb;
    ::ara::chassis::Float32WithValid velocity;
    ::ara::chassis::WheelSpeed wheelSpeedFl;
    ::ara::chassis::WheelSpeed wheelSpeedFr;
    ::ara::chassis::WheelSpeed wheelSpeedRl;
    ::ara::chassis::WheelSpeed wheelSpeedRr;
    ::ara::chassis::Float32WithValid frontWhlIncTarTq;
    ::ara::chassis::Float32WithValid frontWhlDecTarTq;
    ::ara::chassis::Float32WithValid rearWhlIncTarTq;
    ::ara::chassis::Float32WithValid rearWhlDecTarTq;
    ::ara::chassis::Uint8WithValid standstill;
    ::ara::chassis::Float32WithValid roadFriction;
    ::ara::chassis::Float32WithValid brakeTemperature;
    ::ara::chassis::Float32WithValid slope;
    ::UInt8 brakeStatusMaster;
    ::UInt8 brakeStatusSlave;
    ::UInt8 commandFault;
    ::UInt8 epbFaultStatus;
    ::UInt8 epbLockStatus;
    ::UInt8 epbButtonStatus;
    ::UInt8 epbOverride;
    ::UInt8 epbAvl;
    ::Float epbClampForce;
    ::Int32 faultCode;
    ::UInt8 autoHoldActvSts;
    ::Boolean ebdActive;
    ::UInt8 ebdStatus;
    ::Boolean absActive;
    ::UInt8 absStatus;
    ::Boolean escActive;
    ::UInt8 escStatus;
    ::Boolean tcsActive;
    ::UInt8 tcsStatus;
    ::UInt8 arpActvSts;
    ::UInt8 arpFctAvl;
    ::UInt8 arpFailrSts;
    ::UInt8 dbfActvSts;
    ::UInt8 hdcActvSts;
    ::UInt8 hhcActvSts;
    ::UInt8 ebaActvSts;
    ::UInt8 ebaAvl;
    ::UInt8 ebaActvRank;
    ::UInt8 awbAvl;
    ::UInt8 awbActv;
    ::UInt8 espABrkDegStWarning;
    ::UInt8 espABrkDegSt;
    ::UInt8 espPriBrStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(brakePedal);
        fun(brakePedalSwitch);
        fun(driverOverride);
        fun(mastCylinderPressure);
        fun(mastCylinderPosition);
        fun(brakeTorque);
        fun(brakeForce);
        fun(brakeTorqueMax);
        fun(driverDesiredTorque);
        fun(brakeTotalTqReq);
        fun(elecBrkTqReq);
        fun(hydrauBrkTqReq);
        fun(brakeOverHeatStatus);
        fun(aebAvl);
        fun(aebActv);
        fun(prefillAvl);
        fun(prefillActv);
        fun(brkCtrlAvl);
        fun(brkCtrlActv);
        fun(espAbortFb);
        fun(velocity);
        fun(wheelSpeedFl);
        fun(wheelSpeedFr);
        fun(wheelSpeedRl);
        fun(wheelSpeedRr);
        fun(frontWhlIncTarTq);
        fun(frontWhlDecTarTq);
        fun(rearWhlIncTarTq);
        fun(rearWhlDecTarTq);
        fun(standstill);
        fun(roadFriction);
        fun(brakeTemperature);
        fun(slope);
        fun(brakeStatusMaster);
        fun(brakeStatusSlave);
        fun(commandFault);
        fun(epbFaultStatus);
        fun(epbLockStatus);
        fun(epbButtonStatus);
        fun(epbOverride);
        fun(epbAvl);
        fun(epbClampForce);
        fun(faultCode);
        fun(autoHoldActvSts);
        fun(ebdActive);
        fun(ebdStatus);
        fun(absActive);
        fun(absStatus);
        fun(escActive);
        fun(escStatus);
        fun(tcsActive);
        fun(tcsStatus);
        fun(arpActvSts);
        fun(arpFctAvl);
        fun(arpFailrSts);
        fun(dbfActvSts);
        fun(hdcActvSts);
        fun(hhcActvSts);
        fun(ebaActvSts);
        fun(ebaAvl);
        fun(ebaActvRank);
        fun(awbAvl);
        fun(awbActv);
        fun(espABrkDegStWarning);
        fun(espABrkDegSt);
        fun(espPriBrStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(brakePedal);
        fun(brakePedalSwitch);
        fun(driverOverride);
        fun(mastCylinderPressure);
        fun(mastCylinderPosition);
        fun(brakeTorque);
        fun(brakeForce);
        fun(brakeTorqueMax);
        fun(driverDesiredTorque);
        fun(brakeTotalTqReq);
        fun(elecBrkTqReq);
        fun(hydrauBrkTqReq);
        fun(brakeOverHeatStatus);
        fun(aebAvl);
        fun(aebActv);
        fun(prefillAvl);
        fun(prefillActv);
        fun(brkCtrlAvl);
        fun(brkCtrlActv);
        fun(espAbortFb);
        fun(velocity);
        fun(wheelSpeedFl);
        fun(wheelSpeedFr);
        fun(wheelSpeedRl);
        fun(wheelSpeedRr);
        fun(frontWhlIncTarTq);
        fun(frontWhlDecTarTq);
        fun(rearWhlIncTarTq);
        fun(rearWhlDecTarTq);
        fun(standstill);
        fun(roadFriction);
        fun(brakeTemperature);
        fun(slope);
        fun(brakeStatusMaster);
        fun(brakeStatusSlave);
        fun(commandFault);
        fun(epbFaultStatus);
        fun(epbLockStatus);
        fun(epbButtonStatus);
        fun(epbOverride);
        fun(epbAvl);
        fun(epbClampForce);
        fun(faultCode);
        fun(autoHoldActvSts);
        fun(ebdActive);
        fun(ebdStatus);
        fun(absActive);
        fun(absStatus);
        fun(escActive);
        fun(escStatus);
        fun(tcsActive);
        fun(tcsStatus);
        fun(arpActvSts);
        fun(arpFctAvl);
        fun(arpFailrSts);
        fun(dbfActvSts);
        fun(hdcActvSts);
        fun(hhcActvSts);
        fun(ebaActvSts);
        fun(ebaAvl);
        fun(ebaActvRank);
        fun(awbAvl);
        fun(awbActv);
        fun(espABrkDegStWarning);
        fun(espABrkDegSt);
        fun(espPriBrStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("brakePedal", brakePedal);
        fun("brakePedalSwitch", brakePedalSwitch);
        fun("driverOverride", driverOverride);
        fun("mastCylinderPressure", mastCylinderPressure);
        fun("mastCylinderPosition", mastCylinderPosition);
        fun("brakeTorque", brakeTorque);
        fun("brakeForce", brakeForce);
        fun("brakeTorqueMax", brakeTorqueMax);
        fun("driverDesiredTorque", driverDesiredTorque);
        fun("brakeTotalTqReq", brakeTotalTqReq);
        fun("elecBrkTqReq", elecBrkTqReq);
        fun("hydrauBrkTqReq", hydrauBrkTqReq);
        fun("brakeOverHeatStatus", brakeOverHeatStatus);
        fun("aebAvl", aebAvl);
        fun("aebActv", aebActv);
        fun("prefillAvl", prefillAvl);
        fun("prefillActv", prefillActv);
        fun("brkCtrlAvl", brkCtrlAvl);
        fun("brkCtrlActv", brkCtrlActv);
        fun("espAbortFb", espAbortFb);
        fun("velocity", velocity);
        fun("wheelSpeedFl", wheelSpeedFl);
        fun("wheelSpeedFr", wheelSpeedFr);
        fun("wheelSpeedRl", wheelSpeedRl);
        fun("wheelSpeedRr", wheelSpeedRr);
        fun("frontWhlIncTarTq", frontWhlIncTarTq);
        fun("frontWhlDecTarTq", frontWhlDecTarTq);
        fun("rearWhlIncTarTq", rearWhlIncTarTq);
        fun("rearWhlDecTarTq", rearWhlDecTarTq);
        fun("standstill", standstill);
        fun("roadFriction", roadFriction);
        fun("brakeTemperature", brakeTemperature);
        fun("slope", slope);
        fun("brakeStatusMaster", brakeStatusMaster);
        fun("brakeStatusSlave", brakeStatusSlave);
        fun("commandFault", commandFault);
        fun("epbFaultStatus", epbFaultStatus);
        fun("epbLockStatus", epbLockStatus);
        fun("epbButtonStatus", epbButtonStatus);
        fun("epbOverride", epbOverride);
        fun("epbAvl", epbAvl);
        fun("epbClampForce", epbClampForce);
        fun("faultCode", faultCode);
        fun("autoHoldActvSts", autoHoldActvSts);
        fun("ebdActive", ebdActive);
        fun("ebdStatus", ebdStatus);
        fun("absActive", absActive);
        fun("absStatus", absStatus);
        fun("escActive", escActive);
        fun("escStatus", escStatus);
        fun("tcsActive", tcsActive);
        fun("tcsStatus", tcsStatus);
        fun("arpActvSts", arpActvSts);
        fun("arpFctAvl", arpFctAvl);
        fun("arpFailrSts", arpFailrSts);
        fun("dbfActvSts", dbfActvSts);
        fun("hdcActvSts", hdcActvSts);
        fun("hhcActvSts", hhcActvSts);
        fun("ebaActvSts", ebaActvSts);
        fun("ebaAvl", ebaAvl);
        fun("ebaActvRank", ebaActvRank);
        fun("awbAvl", awbAvl);
        fun("awbActv", awbActv);
        fun("espABrkDegStWarning", espABrkDegStWarning);
        fun("espABrkDegSt", espABrkDegSt);
        fun("espPriBrStatus", espPriBrStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("brakePedal", brakePedal);
        fun("brakePedalSwitch", brakePedalSwitch);
        fun("driverOverride", driverOverride);
        fun("mastCylinderPressure", mastCylinderPressure);
        fun("mastCylinderPosition", mastCylinderPosition);
        fun("brakeTorque", brakeTorque);
        fun("brakeForce", brakeForce);
        fun("brakeTorqueMax", brakeTorqueMax);
        fun("driverDesiredTorque", driverDesiredTorque);
        fun("brakeTotalTqReq", brakeTotalTqReq);
        fun("elecBrkTqReq", elecBrkTqReq);
        fun("hydrauBrkTqReq", hydrauBrkTqReq);
        fun("brakeOverHeatStatus", brakeOverHeatStatus);
        fun("aebAvl", aebAvl);
        fun("aebActv", aebActv);
        fun("prefillAvl", prefillAvl);
        fun("prefillActv", prefillActv);
        fun("brkCtrlAvl", brkCtrlAvl);
        fun("brkCtrlActv", brkCtrlActv);
        fun("espAbortFb", espAbortFb);
        fun("velocity", velocity);
        fun("wheelSpeedFl", wheelSpeedFl);
        fun("wheelSpeedFr", wheelSpeedFr);
        fun("wheelSpeedRl", wheelSpeedRl);
        fun("wheelSpeedRr", wheelSpeedRr);
        fun("frontWhlIncTarTq", frontWhlIncTarTq);
        fun("frontWhlDecTarTq", frontWhlDecTarTq);
        fun("rearWhlIncTarTq", rearWhlIncTarTq);
        fun("rearWhlDecTarTq", rearWhlDecTarTq);
        fun("standstill", standstill);
        fun("roadFriction", roadFriction);
        fun("brakeTemperature", brakeTemperature);
        fun("slope", slope);
        fun("brakeStatusMaster", brakeStatusMaster);
        fun("brakeStatusSlave", brakeStatusSlave);
        fun("commandFault", commandFault);
        fun("epbFaultStatus", epbFaultStatus);
        fun("epbLockStatus", epbLockStatus);
        fun("epbButtonStatus", epbButtonStatus);
        fun("epbOverride", epbOverride);
        fun("epbAvl", epbAvl);
        fun("epbClampForce", epbClampForce);
        fun("faultCode", faultCode);
        fun("autoHoldActvSts", autoHoldActvSts);
        fun("ebdActive", ebdActive);
        fun("ebdStatus", ebdStatus);
        fun("absActive", absActive);
        fun("absStatus", absStatus);
        fun("escActive", escActive);
        fun("escStatus", escStatus);
        fun("tcsActive", tcsActive);
        fun("tcsStatus", tcsStatus);
        fun("arpActvSts", arpActvSts);
        fun("arpFctAvl", arpFctAvl);
        fun("arpFailrSts", arpFailrSts);
        fun("dbfActvSts", dbfActvSts);
        fun("hdcActvSts", hdcActvSts);
        fun("hhcActvSts", hhcActvSts);
        fun("ebaActvSts", ebaActvSts);
        fun("ebaAvl", ebaAvl);
        fun("ebaActvRank", ebaActvRank);
        fun("awbAvl", awbAvl);
        fun("awbActv", awbActv);
        fun("espABrkDegStWarning", espABrkDegStWarning);
        fun("espABrkDegSt", espABrkDegSt);
        fun("espPriBrStatus", espPriBrStatus);
    }

    bool operator==(const ::ara::chassis::BrakeInfo& t) const
    {
        return (brakePedal == t.brakePedal) && (brakePedalSwitch == t.brakePedalSwitch) && (driverOverride == t.driverOverride) && (mastCylinderPressure == t.mastCylinderPressure) && (mastCylinderPosition == t.mastCylinderPosition) && (brakeTorque == t.brakeTorque) && (brakeForce == t.brakeForce) && (brakeTorqueMax == t.brakeTorqueMax) && (driverDesiredTorque == t.driverDesiredTorque) && (brakeTotalTqReq == t.brakeTotalTqReq) && (elecBrkTqReq == t.elecBrkTqReq) && (hydrauBrkTqReq == t.hydrauBrkTqReq) && (brakeOverHeatStatus == t.brakeOverHeatStatus) && (aebAvl == t.aebAvl) && (aebActv == t.aebActv) && (prefillAvl == t.prefillAvl) && (prefillActv == t.prefillActv) && (brkCtrlAvl == t.brkCtrlAvl) && (brkCtrlActv == t.brkCtrlActv) && (espAbortFb == t.espAbortFb) && (velocity == t.velocity) && (wheelSpeedFl == t.wheelSpeedFl) && (wheelSpeedFr == t.wheelSpeedFr) && (wheelSpeedRl == t.wheelSpeedRl) && (wheelSpeedRr == t.wheelSpeedRr) && (frontWhlIncTarTq == t.frontWhlIncTarTq) && (frontWhlDecTarTq == t.frontWhlDecTarTq) && (rearWhlIncTarTq == t.rearWhlIncTarTq) && (rearWhlDecTarTq == t.rearWhlDecTarTq) && (standstill == t.standstill) && (roadFriction == t.roadFriction) && (brakeTemperature == t.brakeTemperature) && (slope == t.slope) && (brakeStatusMaster == t.brakeStatusMaster) && (brakeStatusSlave == t.brakeStatusSlave) && (commandFault == t.commandFault) && (epbFaultStatus == t.epbFaultStatus) && (epbLockStatus == t.epbLockStatus) && (epbButtonStatus == t.epbButtonStatus) && (epbOverride == t.epbOverride) && (epbAvl == t.epbAvl) && (fabs(static_cast<double>(epbClampForce - t.epbClampForce)) < DBL_EPSILON) && (faultCode == t.faultCode) && (autoHoldActvSts == t.autoHoldActvSts) && (ebdActive == t.ebdActive) && (ebdStatus == t.ebdStatus) && (absActive == t.absActive) && (absStatus == t.absStatus) && (escActive == t.escActive) && (escStatus == t.escStatus) && (tcsActive == t.tcsActive) && (tcsStatus == t.tcsStatus) && (arpActvSts == t.arpActvSts) && (arpFctAvl == t.arpFctAvl) && (arpFailrSts == t.arpFailrSts) && (dbfActvSts == t.dbfActvSts) && (hdcActvSts == t.hdcActvSts) && (hhcActvSts == t.hhcActvSts) && (ebaActvSts == t.ebaActvSts) && (ebaAvl == t.ebaAvl) && (ebaActvRank == t.ebaActvRank) && (awbAvl == t.awbAvl) && (awbActv == t.awbActv) && (espABrkDegStWarning == t.espABrkDegStWarning) && (espABrkDegSt == t.espABrkDegSt) && (espPriBrStatus == t.espPriBrStatus);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_BRAKEINFO_H
