/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_APAFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_APAFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "ara/vehicle/impl_type_time.h"
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct APAFr01Info {
    ::Boolean valid;
    ::ara::vehicle::Time time;
    ::UInt8 APA_PAS_ObjectSts_FLM;
    ::UInt8 APA_PAS_ObjectSts_FLC;
    ::UInt8 APA_PAS_ObjectSts_FRC;
    ::UInt8 APA_PAS_ObjectSts_FRM;
    ::UInt8 APA_PAS_ObjectSts_RLM;
    ::UInt8 APA_PAS_ObjectSts_RLC;
    ::UInt8 APA_PAS_ObjectSts_PRC;
    ::UInt8 APA_PAS_ObjectSts_RRM;
    ::UInt8 APA_PAS_ObjectSts_SFR;
    ::UInt8 APA_PAS_ObjectSts_SFL;
    ::UInt8 APA_FPAS_WorkSts;
    ::UInt8 APA_PAS_ObjectSts_SRR;
    ::UInt8 APA_PAS_ObjectSts_SRL;
    ::UInt8 APA_RPAS_SensorFaultStsRLM;
    ::UInt8 APA_RPAS_SensorFaultStsRLC;
    ::UInt8 APA_RPAS_SensorFaultStsFRC;
    ::UInt8 APA_RPAS_SensorFaultStsFRM;
    ::UInt8 APA_RPAS_SensorFaultStsFLM;
    ::UInt8 APA_RPAS_SensorFaultStsFLC;
    ::UInt8 APA_RPAS_WorkSts;
    ::UInt8 APA_RPAS_SensorFaultStsSRR;
    ::UInt8 APA_RPAS_SensorFaultStsSRL;
    ::UInt8 APA_RPAS_SensorFaultStsSFR;
    ::UInt8 APA_RPAS_SensorFaultStsSFL;
    ::UInt8 APA_RPAS_SensorFaultStsRRC;
    ::UInt8 APA_RPAS_SensorFaultStsRRM;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(valid);
        fun(time);
        fun(APA_PAS_ObjectSts_FLM);
        fun(APA_PAS_ObjectSts_FLC);
        fun(APA_PAS_ObjectSts_FRC);
        fun(APA_PAS_ObjectSts_FRM);
        fun(APA_PAS_ObjectSts_RLM);
        fun(APA_PAS_ObjectSts_RLC);
        fun(APA_PAS_ObjectSts_PRC);
        fun(APA_PAS_ObjectSts_RRM);
        fun(APA_PAS_ObjectSts_SFR);
        fun(APA_PAS_ObjectSts_SFL);
        fun(APA_FPAS_WorkSts);
        fun(APA_PAS_ObjectSts_SRR);
        fun(APA_PAS_ObjectSts_SRL);
        fun(APA_RPAS_SensorFaultStsRLM);
        fun(APA_RPAS_SensorFaultStsRLC);
        fun(APA_RPAS_SensorFaultStsFRC);
        fun(APA_RPAS_SensorFaultStsFRM);
        fun(APA_RPAS_SensorFaultStsFLM);
        fun(APA_RPAS_SensorFaultStsFLC);
        fun(APA_RPAS_WorkSts);
        fun(APA_RPAS_SensorFaultStsSRR);
        fun(APA_RPAS_SensorFaultStsSRL);
        fun(APA_RPAS_SensorFaultStsSFR);
        fun(APA_RPAS_SensorFaultStsSFL);
        fun(APA_RPAS_SensorFaultStsRRC);
        fun(APA_RPAS_SensorFaultStsRRM);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(valid);
        fun(time);
        fun(APA_PAS_ObjectSts_FLM);
        fun(APA_PAS_ObjectSts_FLC);
        fun(APA_PAS_ObjectSts_FRC);
        fun(APA_PAS_ObjectSts_FRM);
        fun(APA_PAS_ObjectSts_RLM);
        fun(APA_PAS_ObjectSts_RLC);
        fun(APA_PAS_ObjectSts_PRC);
        fun(APA_PAS_ObjectSts_RRM);
        fun(APA_PAS_ObjectSts_SFR);
        fun(APA_PAS_ObjectSts_SFL);
        fun(APA_FPAS_WorkSts);
        fun(APA_PAS_ObjectSts_SRR);
        fun(APA_PAS_ObjectSts_SRL);
        fun(APA_RPAS_SensorFaultStsRLM);
        fun(APA_RPAS_SensorFaultStsRLC);
        fun(APA_RPAS_SensorFaultStsFRC);
        fun(APA_RPAS_SensorFaultStsFRM);
        fun(APA_RPAS_SensorFaultStsFLM);
        fun(APA_RPAS_SensorFaultStsFLC);
        fun(APA_RPAS_WorkSts);
        fun(APA_RPAS_SensorFaultStsSRR);
        fun(APA_RPAS_SensorFaultStsSRL);
        fun(APA_RPAS_SensorFaultStsSFR);
        fun(APA_RPAS_SensorFaultStsSFL);
        fun(APA_RPAS_SensorFaultStsRRC);
        fun(APA_RPAS_SensorFaultStsRRM);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_PAS_ObjectSts_FLM", APA_PAS_ObjectSts_FLM);
        fun("APA_PAS_ObjectSts_FLC", APA_PAS_ObjectSts_FLC);
        fun("APA_PAS_ObjectSts_FRC", APA_PAS_ObjectSts_FRC);
        fun("APA_PAS_ObjectSts_FRM", APA_PAS_ObjectSts_FRM);
        fun("APA_PAS_ObjectSts_RLM", APA_PAS_ObjectSts_RLM);
        fun("APA_PAS_ObjectSts_RLC", APA_PAS_ObjectSts_RLC);
        fun("APA_PAS_ObjectSts_PRC", APA_PAS_ObjectSts_PRC);
        fun("APA_PAS_ObjectSts_RRM", APA_PAS_ObjectSts_RRM);
        fun("APA_PAS_ObjectSts_SFR", APA_PAS_ObjectSts_SFR);
        fun("APA_PAS_ObjectSts_SFL", APA_PAS_ObjectSts_SFL);
        fun("APA_FPAS_WorkSts", APA_FPAS_WorkSts);
        fun("APA_PAS_ObjectSts_SRR", APA_PAS_ObjectSts_SRR);
        fun("APA_PAS_ObjectSts_SRL", APA_PAS_ObjectSts_SRL);
        fun("APA_RPAS_SensorFaultStsRLM", APA_RPAS_SensorFaultStsRLM);
        fun("APA_RPAS_SensorFaultStsRLC", APA_RPAS_SensorFaultStsRLC);
        fun("APA_RPAS_SensorFaultStsFRC", APA_RPAS_SensorFaultStsFRC);
        fun("APA_RPAS_SensorFaultStsFRM", APA_RPAS_SensorFaultStsFRM);
        fun("APA_RPAS_SensorFaultStsFLM", APA_RPAS_SensorFaultStsFLM);
        fun("APA_RPAS_SensorFaultStsFLC", APA_RPAS_SensorFaultStsFLC);
        fun("APA_RPAS_WorkSts", APA_RPAS_WorkSts);
        fun("APA_RPAS_SensorFaultStsSRR", APA_RPAS_SensorFaultStsSRR);
        fun("APA_RPAS_SensorFaultStsSRL", APA_RPAS_SensorFaultStsSRL);
        fun("APA_RPAS_SensorFaultStsSFR", APA_RPAS_SensorFaultStsSFR);
        fun("APA_RPAS_SensorFaultStsSFL", APA_RPAS_SensorFaultStsSFL);
        fun("APA_RPAS_SensorFaultStsRRC", APA_RPAS_SensorFaultStsRRC);
        fun("APA_RPAS_SensorFaultStsRRM", APA_RPAS_SensorFaultStsRRM);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_PAS_ObjectSts_FLM", APA_PAS_ObjectSts_FLM);
        fun("APA_PAS_ObjectSts_FLC", APA_PAS_ObjectSts_FLC);
        fun("APA_PAS_ObjectSts_FRC", APA_PAS_ObjectSts_FRC);
        fun("APA_PAS_ObjectSts_FRM", APA_PAS_ObjectSts_FRM);
        fun("APA_PAS_ObjectSts_RLM", APA_PAS_ObjectSts_RLM);
        fun("APA_PAS_ObjectSts_RLC", APA_PAS_ObjectSts_RLC);
        fun("APA_PAS_ObjectSts_PRC", APA_PAS_ObjectSts_PRC);
        fun("APA_PAS_ObjectSts_RRM", APA_PAS_ObjectSts_RRM);
        fun("APA_PAS_ObjectSts_SFR", APA_PAS_ObjectSts_SFR);
        fun("APA_PAS_ObjectSts_SFL", APA_PAS_ObjectSts_SFL);
        fun("APA_FPAS_WorkSts", APA_FPAS_WorkSts);
        fun("APA_PAS_ObjectSts_SRR", APA_PAS_ObjectSts_SRR);
        fun("APA_PAS_ObjectSts_SRL", APA_PAS_ObjectSts_SRL);
        fun("APA_RPAS_SensorFaultStsRLM", APA_RPAS_SensorFaultStsRLM);
        fun("APA_RPAS_SensorFaultStsRLC", APA_RPAS_SensorFaultStsRLC);
        fun("APA_RPAS_SensorFaultStsFRC", APA_RPAS_SensorFaultStsFRC);
        fun("APA_RPAS_SensorFaultStsFRM", APA_RPAS_SensorFaultStsFRM);
        fun("APA_RPAS_SensorFaultStsFLM", APA_RPAS_SensorFaultStsFLM);
        fun("APA_RPAS_SensorFaultStsFLC", APA_RPAS_SensorFaultStsFLC);
        fun("APA_RPAS_WorkSts", APA_RPAS_WorkSts);
        fun("APA_RPAS_SensorFaultStsSRR", APA_RPAS_SensorFaultStsSRR);
        fun("APA_RPAS_SensorFaultStsSRL", APA_RPAS_SensorFaultStsSRL);
        fun("APA_RPAS_SensorFaultStsSFR", APA_RPAS_SensorFaultStsSFR);
        fun("APA_RPAS_SensorFaultStsSFL", APA_RPAS_SensorFaultStsSFL);
        fun("APA_RPAS_SensorFaultStsRRC", APA_RPAS_SensorFaultStsRRC);
        fun("APA_RPAS_SensorFaultStsRRM", APA_RPAS_SensorFaultStsRRM);
    }

    bool operator==(const ::ara::vehicle::APAFr01Info& t) const
    {
        return (valid == t.valid) && (time == t.time) && (APA_PAS_ObjectSts_FLM == t.APA_PAS_ObjectSts_FLM) && (APA_PAS_ObjectSts_FLC == t.APA_PAS_ObjectSts_FLC) && (APA_PAS_ObjectSts_FRC == t.APA_PAS_ObjectSts_FRC) && (APA_PAS_ObjectSts_FRM == t.APA_PAS_ObjectSts_FRM) && (APA_PAS_ObjectSts_RLM == t.APA_PAS_ObjectSts_RLM) && (APA_PAS_ObjectSts_RLC == t.APA_PAS_ObjectSts_RLC) && (APA_PAS_ObjectSts_PRC == t.APA_PAS_ObjectSts_PRC) && (APA_PAS_ObjectSts_RRM == t.APA_PAS_ObjectSts_RRM) && (APA_PAS_ObjectSts_SFR == t.APA_PAS_ObjectSts_SFR) && (APA_PAS_ObjectSts_SFL == t.APA_PAS_ObjectSts_SFL) && (APA_FPAS_WorkSts == t.APA_FPAS_WorkSts) && (APA_PAS_ObjectSts_SRR == t.APA_PAS_ObjectSts_SRR) && (APA_PAS_ObjectSts_SRL == t.APA_PAS_ObjectSts_SRL) && (APA_RPAS_SensorFaultStsRLM == t.APA_RPAS_SensorFaultStsRLM) && (APA_RPAS_SensorFaultStsRLC == t.APA_RPAS_SensorFaultStsRLC) && (APA_RPAS_SensorFaultStsFRC == t.APA_RPAS_SensorFaultStsFRC) && (APA_RPAS_SensorFaultStsFRM == t.APA_RPAS_SensorFaultStsFRM) && (APA_RPAS_SensorFaultStsFLM == t.APA_RPAS_SensorFaultStsFLM) && (APA_RPAS_SensorFaultStsFLC == t.APA_RPAS_SensorFaultStsFLC) && (APA_RPAS_WorkSts == t.APA_RPAS_WorkSts) && (APA_RPAS_SensorFaultStsSRR == t.APA_RPAS_SensorFaultStsSRR) && (APA_RPAS_SensorFaultStsSRL == t.APA_RPAS_SensorFaultStsSRL) && (APA_RPAS_SensorFaultStsSFR == t.APA_RPAS_SensorFaultStsSFR) && (APA_RPAS_SensorFaultStsSFL == t.APA_RPAS_SensorFaultStsSFL) && (APA_RPAS_SensorFaultStsRRC == t.APA_RPAS_SensorFaultStsRRC) && (APA_RPAS_SensorFaultStsRRM == t.APA_RPAS_SensorFaultStsRRM);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_APAFR01INFO_H
