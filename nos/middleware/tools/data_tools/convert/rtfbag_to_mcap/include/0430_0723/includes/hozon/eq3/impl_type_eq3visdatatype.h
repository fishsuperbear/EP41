/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_EQ3VISDATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_EQ3VISDATATYPE_H
#include <cfloat>
#include <cmath>
#include "hozon/eq3/impl_type_vislaneinformationdatatype.h"
#include "hozon/eq3/impl_type_vislaneneighborrightdatatype.h"
#include "hozon/eq3/impl_type_vislaneneighborleftdatatype.h"
#include "hozon/eq3/impl_type_vislanenearrightindividualdatatype.h"
#include "hozon/eq3/impl_type_vislanenearleftindividualdatatype.h"

namespace hozon {
namespace eq3 {
struct Eq3VisDataType {
    ::hozon::eq3::VisLaneInformationDataType vis_lane_information_data;
    ::hozon::eq3::VisLaneNeighborRightDataType vis_lane_neighbor_right_data;
    ::hozon::eq3::VisLaneNeighborLeftDataType vis_lane_neighbor_left_data;
    ::hozon::eq3::VisLaneNearRightIndividualDataType vis_lane_near_right_individual_data;
    ::hozon::eq3::VisLaneNearLeftIndividualDataType vis_lane_near_left_individual_data;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_lane_information_data);
        fun(vis_lane_neighbor_right_data);
        fun(vis_lane_neighbor_left_data);
        fun(vis_lane_near_right_individual_data);
        fun(vis_lane_near_left_individual_data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_lane_information_data);
        fun(vis_lane_neighbor_right_data);
        fun(vis_lane_neighbor_left_data);
        fun(vis_lane_near_right_individual_data);
        fun(vis_lane_near_left_individual_data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_lane_information_data", vis_lane_information_data);
        fun("vis_lane_neighbor_right_data", vis_lane_neighbor_right_data);
        fun("vis_lane_neighbor_left_data", vis_lane_neighbor_left_data);
        fun("vis_lane_near_right_individual_data", vis_lane_near_right_individual_data);
        fun("vis_lane_near_left_individual_data", vis_lane_near_left_individual_data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_lane_information_data", vis_lane_information_data);
        fun("vis_lane_neighbor_right_data", vis_lane_neighbor_right_data);
        fun("vis_lane_neighbor_left_data", vis_lane_neighbor_left_data);
        fun("vis_lane_near_right_individual_data", vis_lane_near_right_individual_data);
        fun("vis_lane_near_left_individual_data", vis_lane_near_left_individual_data);
    }

    bool operator==(const ::hozon::eq3::Eq3VisDataType& t) const
    {
        return (vis_lane_information_data == t.vis_lane_information_data) && (vis_lane_neighbor_right_data == t.vis_lane_neighbor_right_data) && (vis_lane_neighbor_left_data == t.vis_lane_neighbor_left_data) && (vis_lane_near_right_individual_data == t.vis_lane_near_right_individual_data) && (vis_lane_near_left_individual_data == t.vis_lane_near_left_individual_data);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_EQ3VISDATATYPE_H
