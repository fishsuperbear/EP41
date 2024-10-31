/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FREESPACE_IMPL_TYPE_FREESPACE_H
#define HOZON_FREESPACE_IMPL_TYPE_FREESPACE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_point3darray.h"
#include "hozon/common/impl_type_commontime.h"

namespace hozon {
namespace freespace {
struct FreeSpace {
    ::Int32 spaceSeq;
    ::UInt8 cls;
    ::UInt8 heightType;
    ::UInt8 sensorType;
    ::hozon::composite::Point3DArray freeSpacePointVRF;
    ::hozon::common::CommonTime timeCreation;
    ::hozon::composite::Point3DArray freeSpaceKeyPointVRF;
    bool isLinkObjFusion;
    ::Int32 obstacleId;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(spaceSeq);
        fun(cls);
        fun(heightType);
        fun(sensorType);
        fun(freeSpacePointVRF);
        fun(timeCreation);
        fun(freeSpaceKeyPointVRF);
        fun(isLinkObjFusion);
        fun(obstacleId);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(spaceSeq);
        fun(cls);
        fun(heightType);
        fun(sensorType);
        fun(freeSpacePointVRF);
        fun(timeCreation);
        fun(freeSpaceKeyPointVRF);
        fun(isLinkObjFusion);
        fun(obstacleId);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("spaceSeq", spaceSeq);
        fun("cls", cls);
        fun("heightType", heightType);
        fun("sensorType", sensorType);
        fun("freeSpacePointVRF", freeSpacePointVRF);
        fun("timeCreation", timeCreation);
        fun("freeSpaceKeyPointVRF", freeSpaceKeyPointVRF);
        fun("isLinkObjFusion", isLinkObjFusion);
        fun("obstacleId", obstacleId);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("spaceSeq", spaceSeq);
        fun("cls", cls);
        fun("heightType", heightType);
        fun("sensorType", sensorType);
        fun("freeSpacePointVRF", freeSpacePointVRF);
        fun("timeCreation", timeCreation);
        fun("freeSpaceKeyPointVRF", freeSpaceKeyPointVRF);
        fun("isLinkObjFusion", isLinkObjFusion);
        fun("obstacleId", obstacleId);
    }

    bool operator==(const ::hozon::freespace::FreeSpace& t) const
    {
        return (spaceSeq == t.spaceSeq) && (cls == t.cls) && (heightType == t.heightType) && (sensorType == t.sensorType) && (freeSpacePointVRF == t.freeSpacePointVRF) && (timeCreation == t.timeCreation) && (freeSpaceKeyPointVRF == t.freeSpaceKeyPointVRF) && (isLinkObjFusion == t.isLinkObjFusion) && (obstacleId == t.obstacleId);
    }
};
} // namespace freespace
} // namespace hozon


#endif // HOZON_FREESPACE_IMPL_TYPE_FREESPACE_H
