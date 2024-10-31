/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_INS_IMPL_TYPE_INSINFO_H
#define ARA_INS_IMPL_TYPE_INSINFO_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "impl_type_double.h"
#include "ara/gnss/impl_type_geometrypoit.h"
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"
#include "impl_type_uint32.h"

namespace ara {
namespace ins {
struct InsInfo {
    ::ara::gnss::Header header;
    ::Double longitude;
    ::Double latitude;
    ::Double elevation;
    ::ara::gnss::GeometryPoit utmPosition;
    ::Int32 utmZoneNum;
    ::UInt8 utmZoneChar;
    ::ara::gnss::GeometryPoit attitude;
    ::ara::gnss::GeometryPoit linearVelocity;
    ::ara::gnss::GeometryPoit sdPosition;
    ::ara::gnss::GeometryPoit sdVelocity;
    ::ara::gnss::GeometryPoit sdAttitude;
    ::Double cep68;
    ::Double cep95;
    ::Double second;
    ::Int32 satUseNum;
    ::Int32 satInViewNum;
    ::UInt16 solutionStatus;
    ::UInt16 positionType;
    ::Float pDop;
    ::Float hDop;
    ::Float vDop;
    ::ara::gnss::GeometryPoit attitudeDual;
    ::ara::gnss::GeometryPoit sdAngleDual;
    ::Double baseLineLengthDual;
    ::Int32 solutionStatusDual;
    ::Int32 positionTypeDual;
    ::Int32 solutionSourceDual;
    ::UInt32 aoc;
    ::UInt32 rtkBaseline;
    ::ara::gnss::GeometryPoit angularVelocity;
    ::ara::gnss::GeometryPoit acceleration;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(longitude);
        fun(latitude);
        fun(elevation);
        fun(utmPosition);
        fun(utmZoneNum);
        fun(utmZoneChar);
        fun(attitude);
        fun(linearVelocity);
        fun(sdPosition);
        fun(sdVelocity);
        fun(sdAttitude);
        fun(cep68);
        fun(cep95);
        fun(second);
        fun(satUseNum);
        fun(satInViewNum);
        fun(solutionStatus);
        fun(positionType);
        fun(pDop);
        fun(hDop);
        fun(vDop);
        fun(attitudeDual);
        fun(sdAngleDual);
        fun(baseLineLengthDual);
        fun(solutionStatusDual);
        fun(positionTypeDual);
        fun(solutionSourceDual);
        fun(aoc);
        fun(rtkBaseline);
        fun(angularVelocity);
        fun(acceleration);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(longitude);
        fun(latitude);
        fun(elevation);
        fun(utmPosition);
        fun(utmZoneNum);
        fun(utmZoneChar);
        fun(attitude);
        fun(linearVelocity);
        fun(sdPosition);
        fun(sdVelocity);
        fun(sdAttitude);
        fun(cep68);
        fun(cep95);
        fun(second);
        fun(satUseNum);
        fun(satInViewNum);
        fun(solutionStatus);
        fun(positionType);
        fun(pDop);
        fun(hDop);
        fun(vDop);
        fun(attitudeDual);
        fun(sdAngleDual);
        fun(baseLineLengthDual);
        fun(solutionStatusDual);
        fun(positionTypeDual);
        fun(solutionSourceDual);
        fun(aoc);
        fun(rtkBaseline);
        fun(angularVelocity);
        fun(acceleration);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("longitude", longitude);
        fun("latitude", latitude);
        fun("elevation", elevation);
        fun("utmPosition", utmPosition);
        fun("utmZoneNum", utmZoneNum);
        fun("utmZoneChar", utmZoneChar);
        fun("attitude", attitude);
        fun("linearVelocity", linearVelocity);
        fun("sdPosition", sdPosition);
        fun("sdVelocity", sdVelocity);
        fun("sdAttitude", sdAttitude);
        fun("cep68", cep68);
        fun("cep95", cep95);
        fun("second", second);
        fun("satUseNum", satUseNum);
        fun("satInViewNum", satInViewNum);
        fun("solutionStatus", solutionStatus);
        fun("positionType", positionType);
        fun("pDop", pDop);
        fun("hDop", hDop);
        fun("vDop", vDop);
        fun("attitudeDual", attitudeDual);
        fun("sdAngleDual", sdAngleDual);
        fun("baseLineLengthDual", baseLineLengthDual);
        fun("solutionStatusDual", solutionStatusDual);
        fun("positionTypeDual", positionTypeDual);
        fun("solutionSourceDual", solutionSourceDual);
        fun("aoc", aoc);
        fun("rtkBaseline", rtkBaseline);
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("longitude", longitude);
        fun("latitude", latitude);
        fun("elevation", elevation);
        fun("utmPosition", utmPosition);
        fun("utmZoneNum", utmZoneNum);
        fun("utmZoneChar", utmZoneChar);
        fun("attitude", attitude);
        fun("linearVelocity", linearVelocity);
        fun("sdPosition", sdPosition);
        fun("sdVelocity", sdVelocity);
        fun("sdAttitude", sdAttitude);
        fun("cep68", cep68);
        fun("cep95", cep95);
        fun("second", second);
        fun("satUseNum", satUseNum);
        fun("satInViewNum", satInViewNum);
        fun("solutionStatus", solutionStatus);
        fun("positionType", positionType);
        fun("pDop", pDop);
        fun("hDop", hDop);
        fun("vDop", vDop);
        fun("attitudeDual", attitudeDual);
        fun("sdAngleDual", sdAngleDual);
        fun("baseLineLengthDual", baseLineLengthDual);
        fun("solutionStatusDual", solutionStatusDual);
        fun("positionTypeDual", positionTypeDual);
        fun("solutionSourceDual", solutionSourceDual);
        fun("aoc", aoc);
        fun("rtkBaseline", rtkBaseline);
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
    }

    bool operator==(const ::ara::ins::InsInfo& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(longitude - t.longitude)) < DBL_EPSILON) && (fabs(static_cast<double>(latitude - t.latitude)) < DBL_EPSILON) && (fabs(static_cast<double>(elevation - t.elevation)) < DBL_EPSILON) && (utmPosition == t.utmPosition) && (utmZoneNum == t.utmZoneNum) && (utmZoneChar == t.utmZoneChar) && (attitude == t.attitude) && (linearVelocity == t.linearVelocity) && (sdPosition == t.sdPosition) && (sdVelocity == t.sdVelocity) && (sdAttitude == t.sdAttitude) && (fabs(static_cast<double>(cep68 - t.cep68)) < DBL_EPSILON) && (fabs(static_cast<double>(cep95 - t.cep95)) < DBL_EPSILON) && (fabs(static_cast<double>(second - t.second)) < DBL_EPSILON) && (satUseNum == t.satUseNum) && (satInViewNum == t.satInViewNum) && (solutionStatus == t.solutionStatus) && (positionType == t.positionType) && (fabs(static_cast<double>(pDop - t.pDop)) < DBL_EPSILON) && (fabs(static_cast<double>(hDop - t.hDop)) < DBL_EPSILON) && (fabs(static_cast<double>(vDop - t.vDop)) < DBL_EPSILON) && (attitudeDual == t.attitudeDual) && (sdAngleDual == t.sdAngleDual) && (fabs(static_cast<double>(baseLineLengthDual - t.baseLineLengthDual)) < DBL_EPSILON) && (solutionStatusDual == t.solutionStatusDual) && (positionTypeDual == t.positionTypeDual) && (solutionSourceDual == t.solutionSourceDual) && (aoc == t.aoc) && (rtkBaseline == t.rtkBaseline) && (angularVelocity == t.angularVelocity) && (acceleration == t.acceleration);
    }
};
} // namespace ins
} // namespace ara


#endif // ARA_INS_IMPL_TYPE_INSINFO_H
