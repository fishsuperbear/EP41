/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_NAVIGATIONINFO_H
#define ADSFI_IMPL_TYPE_NAVIGATIONINFO_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_roadpiecevector.h"
#include "impl_type_naviroadpointvector.h"
#include "impl_type_string.h"
#include "impl_type_boolean.h"
#include "impl_type_stringvector.h"
#include "impl_type_lanepiecevector.h"

namespace adsfi {
struct NavigationInfo {
    ::ara::common::CommonHeader header;
    ::RoadPieceVector roadPieces;
    ::NaviRoadPointVector selectPoints;
    ::String mapVersion;
    ::Boolean isReRouted;
    ::StringVector blacklistRoads;
    ::LanePieceVector blacklistLanePieces;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(roadPieces);
        fun(selectPoints);
        fun(mapVersion);
        fun(isReRouted);
        fun(blacklistRoads);
        fun(blacklistLanePieces);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(roadPieces);
        fun(selectPoints);
        fun(mapVersion);
        fun(isReRouted);
        fun(blacklistRoads);
        fun(blacklistLanePieces);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("roadPieces", roadPieces);
        fun("selectPoints", selectPoints);
        fun("mapVersion", mapVersion);
        fun("isReRouted", isReRouted);
        fun("blacklistRoads", blacklistRoads);
        fun("blacklistLanePieces", blacklistLanePieces);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("roadPieces", roadPieces);
        fun("selectPoints", selectPoints);
        fun("mapVersion", mapVersion);
        fun("isReRouted", isReRouted);
        fun("blacklistRoads", blacklistRoads);
        fun("blacklistLanePieces", blacklistLanePieces);
    }

    bool operator==(const ::adsfi::NavigationInfo& t) const
    {
        return (header == t.header) && (roadPieces == t.roadPieces) && (selectPoints == t.selectPoints) && (mapVersion == t.mapVersion) && (isReRouted == t.isReRouted) && (blacklistRoads == t.blacklistRoads) && (blacklistLanePieces == t.blacklistLanePieces);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_NAVIGATIONINFO_H
