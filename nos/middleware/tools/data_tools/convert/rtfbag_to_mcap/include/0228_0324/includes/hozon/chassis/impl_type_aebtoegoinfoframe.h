/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_AEBTOEGOINFOFRAME_H
#define HOZON_CHASSIS_IMPL_TYPE_AEBTOEGOINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AebToEgoInfoFrame {
    ::hozon::common::CommonHeader header;
    ::uint8_t AEB_state;
    ::uint8_t AEB_target_id;
    ::uint8_t FCW_state;
    ::uint8_t FCW_target_id;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(AEB_state);
        fun(AEB_target_id);
        fun(FCW_state);
        fun(FCW_target_id);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(AEB_state);
        fun(AEB_target_id);
        fun(FCW_state);
        fun(FCW_target_id);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("AEB_state", AEB_state);
        fun("AEB_target_id", AEB_target_id);
        fun("FCW_state", FCW_state);
        fun("FCW_target_id", FCW_target_id);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("AEB_state", AEB_state);
        fun("AEB_target_id", AEB_target_id);
        fun("FCW_state", FCW_state);
        fun("FCW_target_id", FCW_target_id);
    }

    bool operator==(const ::hozon::chassis::AebToEgoInfoFrame& t) const
    {
        return (header == t.header) && (AEB_state == t.AEB_state) && (AEB_target_id == t.AEB_target_id) && (FCW_state == t.FCW_state) && (FCW_target_id == t.FCW_target_id);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_AEBTOEGOINFOFRAME_H
