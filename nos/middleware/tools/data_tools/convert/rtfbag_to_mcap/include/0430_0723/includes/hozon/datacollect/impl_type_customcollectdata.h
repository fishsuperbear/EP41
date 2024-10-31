/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DATACOLLECT_IMPL_TYPE_CUSTOMCOLLECTDATA_H
#define HOZON_DATACOLLECT_IMPL_TYPE_CUSTOMCOLLECTDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/datacollect/impl_type_customrawdata.h"

namespace hozon {
namespace datacollect {
struct CustomCollectData {
    ::UInt32 data_type;
    ::hozon::datacollect::CustomRawData raw_data;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(data_type);
        fun(raw_data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(data_type);
        fun(raw_data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("data_type", data_type);
        fun("raw_data", raw_data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("data_type", data_type);
        fun("raw_data", raw_data);
    }

    bool operator==(const ::hozon::datacollect::CustomCollectData& t) const
    {
        return (data_type == t.data_type) && (raw_data == t.raw_data);
    }
};
} // namespace datacollect
} // namespace hozon


#endif // HOZON_DATACOLLECT_IMPL_TYPE_CUSTOMCOLLECTDATA_H
