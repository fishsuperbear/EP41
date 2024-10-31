/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HMIHAFPLANNINGDECISIONINFO_H
#define HOZON_HMI_IMPL_TYPE_HMIHAFPLANNINGDECISIONINFO_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/hmi/impl_type_hmidecisioninfo.h"
#include "impl_type_boolean.h"
#include "hozon/planning/impl_type_egotrajectoryframe.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace hmi {
struct HmiHafPlanningDecisionInfo {
    ::hozon::common::CommonHeader header;
    ::hozon::hmi::HmiDecisionInfo decisionInfo;
    ::Boolean IsValid;
    ::hozon::planning::EgoTrajectoryFrame ego;
    ::UInt8 is_nnp_active;
    ::UInt8 spd_limit_hd_map;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(decisionInfo);
        fun(IsValid);
        fun(ego);
        fun(is_nnp_active);
        fun(spd_limit_hd_map);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(decisionInfo);
        fun(IsValid);
        fun(ego);
        fun(is_nnp_active);
        fun(spd_limit_hd_map);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("decisionInfo", decisionInfo);
        fun("IsValid", IsValid);
        fun("ego", ego);
        fun("is_nnp_active", is_nnp_active);
        fun("spd_limit_hd_map", spd_limit_hd_map);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("decisionInfo", decisionInfo);
        fun("IsValid", IsValid);
        fun("ego", ego);
        fun("is_nnp_active", is_nnp_active);
        fun("spd_limit_hd_map", spd_limit_hd_map);
    }

    bool operator==(const ::hozon::hmi::HmiHafPlanningDecisionInfo& t) const
    {
        return (header == t.header) && (decisionInfo == t.decisionInfo) && (IsValid == t.IsValid) && (ego == t.ego) && (is_nnp_active == t.is_nnp_active) && (spd_limit_hd_map == t.spd_limit_hd_map);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HMIHAFPLANNINGDECISIONINFO_H
