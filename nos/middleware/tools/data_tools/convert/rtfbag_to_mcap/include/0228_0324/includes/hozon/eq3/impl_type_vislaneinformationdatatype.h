/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_VISLANEINFORMATIONDATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_VISLANEINFORMATIONDATATYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace eq3 {
struct VisLaneInformationDataType {
    ::uint8_t vis_lane_left_individ_type;
    ::uint8_t vis_lane_left_parall_type;
    ::uint8_t vis_lane_right_individ_type;
    ::uint8_t vis_lane_right_parall_type;
    ::uint8_t vis_lane_left_parall_dimonconf;
    ::uint8_t vis_lane_left_parall_lkaconf;
    ::uint8_t vis_lane_left_parall_tjaconf;
    ::uint8_t vis_lane_left_individ_dimonconf;
    ::uint8_t vis_lane_left_individ_lkaconf;
    ::uint8_t vis_lane_left_individ_tjaconf;
    ::uint8_t vis_lane_right_parall_dimonconf;
    ::uint8_t vis_lane_right_parall_lkaconf;
    ::uint8_t vis_lane_right_parall_tjaconf;
    ::uint8_t vis_lane_right_individ_dimonconf;
    ::uint8_t vis_lane_right_individ_lkaconf;
    ::uint8_t vis_lane_right_individ_tjaconf;
    ::uint8_t vis_lane_lane_change;
    ::uint8_t vis_lane_ambiguous_lane_left;
    ::uint8_t vis_lane_ambiguous_lane_right;
    ::uint8_t vis_road_istunnel_entryexit;
    ::uint8_t vis_lane_parall_prob;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_lane_left_individ_type);
        fun(vis_lane_left_parall_type);
        fun(vis_lane_right_individ_type);
        fun(vis_lane_right_parall_type);
        fun(vis_lane_left_parall_dimonconf);
        fun(vis_lane_left_parall_lkaconf);
        fun(vis_lane_left_parall_tjaconf);
        fun(vis_lane_left_individ_dimonconf);
        fun(vis_lane_left_individ_lkaconf);
        fun(vis_lane_left_individ_tjaconf);
        fun(vis_lane_right_parall_dimonconf);
        fun(vis_lane_right_parall_lkaconf);
        fun(vis_lane_right_parall_tjaconf);
        fun(vis_lane_right_individ_dimonconf);
        fun(vis_lane_right_individ_lkaconf);
        fun(vis_lane_right_individ_tjaconf);
        fun(vis_lane_lane_change);
        fun(vis_lane_ambiguous_lane_left);
        fun(vis_lane_ambiguous_lane_right);
        fun(vis_road_istunnel_entryexit);
        fun(vis_lane_parall_prob);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_lane_left_individ_type);
        fun(vis_lane_left_parall_type);
        fun(vis_lane_right_individ_type);
        fun(vis_lane_right_parall_type);
        fun(vis_lane_left_parall_dimonconf);
        fun(vis_lane_left_parall_lkaconf);
        fun(vis_lane_left_parall_tjaconf);
        fun(vis_lane_left_individ_dimonconf);
        fun(vis_lane_left_individ_lkaconf);
        fun(vis_lane_left_individ_tjaconf);
        fun(vis_lane_right_parall_dimonconf);
        fun(vis_lane_right_parall_lkaconf);
        fun(vis_lane_right_parall_tjaconf);
        fun(vis_lane_right_individ_dimonconf);
        fun(vis_lane_right_individ_lkaconf);
        fun(vis_lane_right_individ_tjaconf);
        fun(vis_lane_lane_change);
        fun(vis_lane_ambiguous_lane_left);
        fun(vis_lane_ambiguous_lane_right);
        fun(vis_road_istunnel_entryexit);
        fun(vis_lane_parall_prob);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_lane_left_individ_type", vis_lane_left_individ_type);
        fun("vis_lane_left_parall_type", vis_lane_left_parall_type);
        fun("vis_lane_right_individ_type", vis_lane_right_individ_type);
        fun("vis_lane_right_parall_type", vis_lane_right_parall_type);
        fun("vis_lane_left_parall_dimonconf", vis_lane_left_parall_dimonconf);
        fun("vis_lane_left_parall_lkaconf", vis_lane_left_parall_lkaconf);
        fun("vis_lane_left_parall_tjaconf", vis_lane_left_parall_tjaconf);
        fun("vis_lane_left_individ_dimonconf", vis_lane_left_individ_dimonconf);
        fun("vis_lane_left_individ_lkaconf", vis_lane_left_individ_lkaconf);
        fun("vis_lane_left_individ_tjaconf", vis_lane_left_individ_tjaconf);
        fun("vis_lane_right_parall_dimonconf", vis_lane_right_parall_dimonconf);
        fun("vis_lane_right_parall_lkaconf", vis_lane_right_parall_lkaconf);
        fun("vis_lane_right_parall_tjaconf", vis_lane_right_parall_tjaconf);
        fun("vis_lane_right_individ_dimonconf", vis_lane_right_individ_dimonconf);
        fun("vis_lane_right_individ_lkaconf", vis_lane_right_individ_lkaconf);
        fun("vis_lane_right_individ_tjaconf", vis_lane_right_individ_tjaconf);
        fun("vis_lane_lane_change", vis_lane_lane_change);
        fun("vis_lane_ambiguous_lane_left", vis_lane_ambiguous_lane_left);
        fun("vis_lane_ambiguous_lane_right", vis_lane_ambiguous_lane_right);
        fun("vis_road_istunnel_entryexit", vis_road_istunnel_entryexit);
        fun("vis_lane_parall_prob", vis_lane_parall_prob);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_lane_left_individ_type", vis_lane_left_individ_type);
        fun("vis_lane_left_parall_type", vis_lane_left_parall_type);
        fun("vis_lane_right_individ_type", vis_lane_right_individ_type);
        fun("vis_lane_right_parall_type", vis_lane_right_parall_type);
        fun("vis_lane_left_parall_dimonconf", vis_lane_left_parall_dimonconf);
        fun("vis_lane_left_parall_lkaconf", vis_lane_left_parall_lkaconf);
        fun("vis_lane_left_parall_tjaconf", vis_lane_left_parall_tjaconf);
        fun("vis_lane_left_individ_dimonconf", vis_lane_left_individ_dimonconf);
        fun("vis_lane_left_individ_lkaconf", vis_lane_left_individ_lkaconf);
        fun("vis_lane_left_individ_tjaconf", vis_lane_left_individ_tjaconf);
        fun("vis_lane_right_parall_dimonconf", vis_lane_right_parall_dimonconf);
        fun("vis_lane_right_parall_lkaconf", vis_lane_right_parall_lkaconf);
        fun("vis_lane_right_parall_tjaconf", vis_lane_right_parall_tjaconf);
        fun("vis_lane_right_individ_dimonconf", vis_lane_right_individ_dimonconf);
        fun("vis_lane_right_individ_lkaconf", vis_lane_right_individ_lkaconf);
        fun("vis_lane_right_individ_tjaconf", vis_lane_right_individ_tjaconf);
        fun("vis_lane_lane_change", vis_lane_lane_change);
        fun("vis_lane_ambiguous_lane_left", vis_lane_ambiguous_lane_left);
        fun("vis_lane_ambiguous_lane_right", vis_lane_ambiguous_lane_right);
        fun("vis_road_istunnel_entryexit", vis_road_istunnel_entryexit);
        fun("vis_lane_parall_prob", vis_lane_parall_prob);
    }

    bool operator==(const ::hozon::eq3::VisLaneInformationDataType& t) const
    {
        return (vis_lane_left_individ_type == t.vis_lane_left_individ_type) && (vis_lane_left_parall_type == t.vis_lane_left_parall_type) && (vis_lane_right_individ_type == t.vis_lane_right_individ_type) && (vis_lane_right_parall_type == t.vis_lane_right_parall_type) && (vis_lane_left_parall_dimonconf == t.vis_lane_left_parall_dimonconf) && (vis_lane_left_parall_lkaconf == t.vis_lane_left_parall_lkaconf) && (vis_lane_left_parall_tjaconf == t.vis_lane_left_parall_tjaconf) && (vis_lane_left_individ_dimonconf == t.vis_lane_left_individ_dimonconf) && (vis_lane_left_individ_lkaconf == t.vis_lane_left_individ_lkaconf) && (vis_lane_left_individ_tjaconf == t.vis_lane_left_individ_tjaconf) && (vis_lane_right_parall_dimonconf == t.vis_lane_right_parall_dimonconf) && (vis_lane_right_parall_lkaconf == t.vis_lane_right_parall_lkaconf) && (vis_lane_right_parall_tjaconf == t.vis_lane_right_parall_tjaconf) && (vis_lane_right_individ_dimonconf == t.vis_lane_right_individ_dimonconf) && (vis_lane_right_individ_lkaconf == t.vis_lane_right_individ_lkaconf) && (vis_lane_right_individ_tjaconf == t.vis_lane_right_individ_tjaconf) && (vis_lane_lane_change == t.vis_lane_lane_change) && (vis_lane_ambiguous_lane_left == t.vis_lane_ambiguous_lane_left) && (vis_lane_ambiguous_lane_right == t.vis_lane_ambiguous_lane_right) && (vis_road_istunnel_entryexit == t.vis_road_istunnel_entryexit) && (vis_lane_parall_prob == t.vis_lane_parall_prob);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_VISLANEINFORMATIONDATATYPE_H
