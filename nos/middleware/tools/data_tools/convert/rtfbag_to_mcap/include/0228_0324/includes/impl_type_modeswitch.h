/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_MODESWITCH_H
#define IMPL_TYPE_MODESWITCH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct ModeSwitch {
    ::UInt8 autoDriveSwitch;
    ::UInt8 emergencySwitch;
    ::UInt8 apaSwitch;
    ::UInt8 lkaSwitch;
    ::UInt8 speedLimitSwitch;
    ::UInt8 accSwitchOnOff;
    ::UInt8 accSwitchResume;
    ::UInt8 accSwitchCancel;
    ::UInt8 accSwitchSpeedInc;
    ::UInt8 accSwitchSpeedDec;
    ::UInt8 accSwitchGapInc;
    ::UInt8 accSwitchGapDec;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(autoDriveSwitch);
        fun(emergencySwitch);
        fun(apaSwitch);
        fun(lkaSwitch);
        fun(speedLimitSwitch);
        fun(accSwitchOnOff);
        fun(accSwitchResume);
        fun(accSwitchCancel);
        fun(accSwitchSpeedInc);
        fun(accSwitchSpeedDec);
        fun(accSwitchGapInc);
        fun(accSwitchGapDec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(autoDriveSwitch);
        fun(emergencySwitch);
        fun(apaSwitch);
        fun(lkaSwitch);
        fun(speedLimitSwitch);
        fun(accSwitchOnOff);
        fun(accSwitchResume);
        fun(accSwitchCancel);
        fun(accSwitchSpeedInc);
        fun(accSwitchSpeedDec);
        fun(accSwitchGapInc);
        fun(accSwitchGapDec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("autoDriveSwitch", autoDriveSwitch);
        fun("emergencySwitch", emergencySwitch);
        fun("apaSwitch", apaSwitch);
        fun("lkaSwitch", lkaSwitch);
        fun("speedLimitSwitch", speedLimitSwitch);
        fun("accSwitchOnOff", accSwitchOnOff);
        fun("accSwitchResume", accSwitchResume);
        fun("accSwitchCancel", accSwitchCancel);
        fun("accSwitchSpeedInc", accSwitchSpeedInc);
        fun("accSwitchSpeedDec", accSwitchSpeedDec);
        fun("accSwitchGapInc", accSwitchGapInc);
        fun("accSwitchGapDec", accSwitchGapDec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("autoDriveSwitch", autoDriveSwitch);
        fun("emergencySwitch", emergencySwitch);
        fun("apaSwitch", apaSwitch);
        fun("lkaSwitch", lkaSwitch);
        fun("speedLimitSwitch", speedLimitSwitch);
        fun("accSwitchOnOff", accSwitchOnOff);
        fun("accSwitchResume", accSwitchResume);
        fun("accSwitchCancel", accSwitchCancel);
        fun("accSwitchSpeedInc", accSwitchSpeedInc);
        fun("accSwitchSpeedDec", accSwitchSpeedDec);
        fun("accSwitchGapInc", accSwitchGapInc);
        fun("accSwitchGapDec", accSwitchGapDec);
    }

    bool operator==(const ::ModeSwitch& t) const
    {
        return (autoDriveSwitch == t.autoDriveSwitch) && (emergencySwitch == t.emergencySwitch) && (apaSwitch == t.apaSwitch) && (lkaSwitch == t.lkaSwitch) && (speedLimitSwitch == t.speedLimitSwitch) && (accSwitchOnOff == t.accSwitchOnOff) && (accSwitchResume == t.accSwitchResume) && (accSwitchCancel == t.accSwitchCancel) && (accSwitchSpeedInc == t.accSwitchSpeedInc) && (accSwitchSpeedDec == t.accSwitchSpeedDec) && (accSwitchGapInc == t.accSwitchGapInc) && (accSwitchGapDec == t.accSwitchGapDec);
    }
};


#endif // IMPL_TYPE_MODESWITCH_H
