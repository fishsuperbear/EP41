/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RTRACK_IMPL_TYPE_RADARTRACK_H
#define ARA_RTRACK_IMPL_TYPE_RADARTRACK_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace rtrack {
struct RadarTrack {
    ::UInt8 id;
    ::UInt8 idState;
    ::Float lifetime;
    ::Float x;
    ::Float y;
    ::Float z;
    ::Float vx;
    ::Float vy;
    ::Float ax;
    ::Float ay;
    ::Float rcs;
    ::Float snr;
    ::Float xRms;
    ::Float yRms;
    ::Float zRms;
    ::Float vxRms;
    ::Float vyRms;
    ::Float axRms;
    ::Float ayRms;
    ::Float orientation;
    ::Float orientRms;
    ::Float yawRate;
    ::Float heading;
    ::Float length;
    ::Float width;
    ::Float height;
    ::UInt8 fusionSig;
    ::UInt8 fusionCamID;
    ::Float yawRateRms;
    ::Float lengthRms;
    ::Float widthRms;
    ::Float heightRms;
    ::Float xQuality;
    ::Float yQuality;
    ::Float zQuality;
    ::Float vxQuality;
    ::Float vyQuality;
    ::Float axQuality;
    ::Float ayQuality;
    ::Float orientationQuality;
    ::Float yawRateQuality;
    ::Float lengthQuality;
    ::Float widthQuality;
    ::Float heightQuality;
    ::Float underpassProbability;
    ::Float overpassProbability;
    ::UInt8 existProbability;
    ::UInt8 movProperty;
    ::UInt8 trackState;
    ::UInt8 trackType;
    ::UInt8 referencePoint;
    ::UInt8 motionDirection;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(idState);
        fun(lifetime);
        fun(x);
        fun(y);
        fun(z);
        fun(vx);
        fun(vy);
        fun(ax);
        fun(ay);
        fun(rcs);
        fun(snr);
        fun(xRms);
        fun(yRms);
        fun(zRms);
        fun(vxRms);
        fun(vyRms);
        fun(axRms);
        fun(ayRms);
        fun(orientation);
        fun(orientRms);
        fun(yawRate);
        fun(heading);
        fun(length);
        fun(width);
        fun(height);
        fun(fusionSig);
        fun(fusionCamID);
        fun(yawRateRms);
        fun(lengthRms);
        fun(widthRms);
        fun(heightRms);
        fun(xQuality);
        fun(yQuality);
        fun(zQuality);
        fun(vxQuality);
        fun(vyQuality);
        fun(axQuality);
        fun(ayQuality);
        fun(orientationQuality);
        fun(yawRateQuality);
        fun(lengthQuality);
        fun(widthQuality);
        fun(heightQuality);
        fun(underpassProbability);
        fun(overpassProbability);
        fun(existProbability);
        fun(movProperty);
        fun(trackState);
        fun(trackType);
        fun(referencePoint);
        fun(motionDirection);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(idState);
        fun(lifetime);
        fun(x);
        fun(y);
        fun(z);
        fun(vx);
        fun(vy);
        fun(ax);
        fun(ay);
        fun(rcs);
        fun(snr);
        fun(xRms);
        fun(yRms);
        fun(zRms);
        fun(vxRms);
        fun(vyRms);
        fun(axRms);
        fun(ayRms);
        fun(orientation);
        fun(orientRms);
        fun(yawRate);
        fun(heading);
        fun(length);
        fun(width);
        fun(height);
        fun(fusionSig);
        fun(fusionCamID);
        fun(yawRateRms);
        fun(lengthRms);
        fun(widthRms);
        fun(heightRms);
        fun(xQuality);
        fun(yQuality);
        fun(zQuality);
        fun(vxQuality);
        fun(vyQuality);
        fun(axQuality);
        fun(ayQuality);
        fun(orientationQuality);
        fun(yawRateQuality);
        fun(lengthQuality);
        fun(widthQuality);
        fun(heightQuality);
        fun(underpassProbability);
        fun(overpassProbability);
        fun(existProbability);
        fun(movProperty);
        fun(trackState);
        fun(trackType);
        fun(referencePoint);
        fun(motionDirection);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("idState", idState);
        fun("lifetime", lifetime);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("vx", vx);
        fun("vy", vy);
        fun("ax", ax);
        fun("ay", ay);
        fun("rcs", rcs);
        fun("snr", snr);
        fun("xRms", xRms);
        fun("yRms", yRms);
        fun("zRms", zRms);
        fun("vxRms", vxRms);
        fun("vyRms", vyRms);
        fun("axRms", axRms);
        fun("ayRms", ayRms);
        fun("orientation", orientation);
        fun("orientRms", orientRms);
        fun("yawRate", yawRate);
        fun("heading", heading);
        fun("length", length);
        fun("width", width);
        fun("height", height);
        fun("fusionSig", fusionSig);
        fun("fusionCamID", fusionCamID);
        fun("yawRateRms", yawRateRms);
        fun("lengthRms", lengthRms);
        fun("widthRms", widthRms);
        fun("heightRms", heightRms);
        fun("xQuality", xQuality);
        fun("yQuality", yQuality);
        fun("zQuality", zQuality);
        fun("vxQuality", vxQuality);
        fun("vyQuality", vyQuality);
        fun("axQuality", axQuality);
        fun("ayQuality", ayQuality);
        fun("orientationQuality", orientationQuality);
        fun("yawRateQuality", yawRateQuality);
        fun("lengthQuality", lengthQuality);
        fun("widthQuality", widthQuality);
        fun("heightQuality", heightQuality);
        fun("underpassProbability", underpassProbability);
        fun("overpassProbability", overpassProbability);
        fun("existProbability", existProbability);
        fun("movProperty", movProperty);
        fun("trackState", trackState);
        fun("trackType", trackType);
        fun("referencePoint", referencePoint);
        fun("motionDirection", motionDirection);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("idState", idState);
        fun("lifetime", lifetime);
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("vx", vx);
        fun("vy", vy);
        fun("ax", ax);
        fun("ay", ay);
        fun("rcs", rcs);
        fun("snr", snr);
        fun("xRms", xRms);
        fun("yRms", yRms);
        fun("zRms", zRms);
        fun("vxRms", vxRms);
        fun("vyRms", vyRms);
        fun("axRms", axRms);
        fun("ayRms", ayRms);
        fun("orientation", orientation);
        fun("orientRms", orientRms);
        fun("yawRate", yawRate);
        fun("heading", heading);
        fun("length", length);
        fun("width", width);
        fun("height", height);
        fun("fusionSig", fusionSig);
        fun("fusionCamID", fusionCamID);
        fun("yawRateRms", yawRateRms);
        fun("lengthRms", lengthRms);
        fun("widthRms", widthRms);
        fun("heightRms", heightRms);
        fun("xQuality", xQuality);
        fun("yQuality", yQuality);
        fun("zQuality", zQuality);
        fun("vxQuality", vxQuality);
        fun("vyQuality", vyQuality);
        fun("axQuality", axQuality);
        fun("ayQuality", ayQuality);
        fun("orientationQuality", orientationQuality);
        fun("yawRateQuality", yawRateQuality);
        fun("lengthQuality", lengthQuality);
        fun("widthQuality", widthQuality);
        fun("heightQuality", heightQuality);
        fun("underpassProbability", underpassProbability);
        fun("overpassProbability", overpassProbability);
        fun("existProbability", existProbability);
        fun("movProperty", movProperty);
        fun("trackState", trackState);
        fun("trackType", trackType);
        fun("referencePoint", referencePoint);
        fun("motionDirection", motionDirection);
    }

    bool operator==(const ::ara::rtrack::RadarTrack& t) const
    {
        return (id == t.id) && (idState == t.idState) && (fabs(static_cast<double>(lifetime - t.lifetime)) < DBL_EPSILON) && (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (fabs(static_cast<double>(vx - t.vx)) < DBL_EPSILON) && (fabs(static_cast<double>(vy - t.vy)) < DBL_EPSILON) && (fabs(static_cast<double>(ax - t.ax)) < DBL_EPSILON) && (fabs(static_cast<double>(ay - t.ay)) < DBL_EPSILON) && (fabs(static_cast<double>(rcs - t.rcs)) < DBL_EPSILON) && (fabs(static_cast<double>(snr - t.snr)) < DBL_EPSILON) && (fabs(static_cast<double>(xRms - t.xRms)) < DBL_EPSILON) && (fabs(static_cast<double>(yRms - t.yRms)) < DBL_EPSILON) && (fabs(static_cast<double>(zRms - t.zRms)) < DBL_EPSILON) && (fabs(static_cast<double>(vxRms - t.vxRms)) < DBL_EPSILON) && (fabs(static_cast<double>(vyRms - t.vyRms)) < DBL_EPSILON) && (fabs(static_cast<double>(axRms - t.axRms)) < DBL_EPSILON) && (fabs(static_cast<double>(ayRms - t.ayRms)) < DBL_EPSILON) && (fabs(static_cast<double>(orientation - t.orientation)) < DBL_EPSILON) && (fabs(static_cast<double>(orientRms - t.orientRms)) < DBL_EPSILON) && (fabs(static_cast<double>(yawRate - t.yawRate)) < DBL_EPSILON) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON) && (fabs(static_cast<double>(length - t.length)) < DBL_EPSILON) && (fabs(static_cast<double>(width - t.width)) < DBL_EPSILON) && (fabs(static_cast<double>(height - t.height)) < DBL_EPSILON) && (fusionSig == t.fusionSig) && (fusionCamID == t.fusionCamID) && (fabs(static_cast<double>(yawRateRms - t.yawRateRms)) < DBL_EPSILON) && (fabs(static_cast<double>(lengthRms - t.lengthRms)) < DBL_EPSILON) && (fabs(static_cast<double>(widthRms - t.widthRms)) < DBL_EPSILON) && (fabs(static_cast<double>(heightRms - t.heightRms)) < DBL_EPSILON) && (fabs(static_cast<double>(xQuality - t.xQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(yQuality - t.yQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(zQuality - t.zQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(vxQuality - t.vxQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(vyQuality - t.vyQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(axQuality - t.axQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(ayQuality - t.ayQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(orientationQuality - t.orientationQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(yawRateQuality - t.yawRateQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(lengthQuality - t.lengthQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(widthQuality - t.widthQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(heightQuality - t.heightQuality)) < DBL_EPSILON) && (fabs(static_cast<double>(underpassProbability - t.underpassProbability)) < DBL_EPSILON) && (fabs(static_cast<double>(overpassProbability - t.overpassProbability)) < DBL_EPSILON) && (existProbability == t.existProbability) && (movProperty == t.movProperty) && (trackState == t.trackState) && (trackType == t.trackType) && (referencePoint == t.referencePoint) && (motionDirection == t.motionDirection);
    }
};
} // namespace rtrack
} // namespace ara


#endif // ARA_RTRACK_IMPL_TYPE_RADARTRACK_H
