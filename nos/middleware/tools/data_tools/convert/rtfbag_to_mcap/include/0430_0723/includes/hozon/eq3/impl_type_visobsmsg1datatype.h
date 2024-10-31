/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_VISOBSMSG1DATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_VISOBSMSG1DATATYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace eq3 {
struct VisObsMsg1DataType {
    ::uint8_t vis_obs_id_01;
    ::uint8_t vis_obs_classification_01;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_obs_id_01);
        fun(vis_obs_classification_01);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_obs_id_01);
        fun(vis_obs_classification_01);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_obs_id_01", vis_obs_id_01);
        fun("vis_obs_classification_01", vis_obs_classification_01);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_obs_id_01", vis_obs_id_01);
        fun("vis_obs_classification_01", vis_obs_classification_01);
    }

    bool operator==(const ::hozon::eq3::VisObsMsg1DataType& t) const
    {
        return (vis_obs_id_01 == t.vis_obs_id_01) && (vis_obs_classification_01 == t.vis_obs_classification_01);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_VISOBSMSG1DATATYPE_H
