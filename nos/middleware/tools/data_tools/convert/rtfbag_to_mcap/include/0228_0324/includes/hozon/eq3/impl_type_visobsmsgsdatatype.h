/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_VISOBSMSGSDATATYPE_H
#define HOZON_EQ3_IMPL_TYPE_VISOBSMSGSDATATYPE_H
#include <cfloat>
#include <cmath>
#include "hozon/eq3/impl_type_visobsmsg1datatype.h"
#include "hozon/eq3/impl_type_visobsmsg2datatype.h"
#include "hozon/eq3/impl_type_visobsmsg3datatype.h"

namespace hozon {
namespace eq3 {
struct VisObsMsgsDataType {
    ::hozon::eq3::VisObsMsg1DataType vis_obs_msg1_data;
    ::hozon::eq3::VisObsMsg2DataType vis_obs_msg2_data;
    ::hozon::eq3::VisObsMsg3DataType vis_obs_msg3_data;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vis_obs_msg1_data);
        fun(vis_obs_msg2_data);
        fun(vis_obs_msg3_data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vis_obs_msg1_data);
        fun(vis_obs_msg2_data);
        fun(vis_obs_msg3_data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vis_obs_msg1_data", vis_obs_msg1_data);
        fun("vis_obs_msg2_data", vis_obs_msg2_data);
        fun("vis_obs_msg3_data", vis_obs_msg3_data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vis_obs_msg1_data", vis_obs_msg1_data);
        fun("vis_obs_msg2_data", vis_obs_msg2_data);
        fun("vis_obs_msg3_data", vis_obs_msg3_data);
    }

    bool operator==(const ::hozon::eq3::VisObsMsgsDataType& t) const
    {
        return (vis_obs_msg1_data == t.vis_obs_msg1_data) && (vis_obs_msg2_data == t.vis_obs_msg2_data) && (vis_obs_msg3_data == t.vis_obs_msg3_data);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_VISOBSMSGSDATATYPE_H
