/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGEGOTOMCUFRAME_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGEGOTOMCUFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/chassis/impl_type_algegomcunnpmsg.h"
#include "hozon/chassis/impl_type_algegomcuavpmsg.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace chassis {
struct AlgEgoToMcuFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::chassis::AlgEgoMcuNNPMsg msg_ego_nnp;
    ::hozon::chassis::AlgEgoMcuAVPMsg msg_ego_avp;
    ::UInt32 SOC2FCT_TBD_u32_01;
    ::UInt32 SOC2FCT_TBD_u32_02;
    ::UInt32 SOC2FCT_TBD_u32_03;
    ::UInt32 SOC2FCT_TBD_u32_04;
    ::UInt32 SOC2FCT_TBD_u32_05;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(msg_ego_nnp);
        fun(msg_ego_avp);
        fun(SOC2FCT_TBD_u32_01);
        fun(SOC2FCT_TBD_u32_02);
        fun(SOC2FCT_TBD_u32_03);
        fun(SOC2FCT_TBD_u32_04);
        fun(SOC2FCT_TBD_u32_05);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(msg_ego_nnp);
        fun(msg_ego_avp);
        fun(SOC2FCT_TBD_u32_01);
        fun(SOC2FCT_TBD_u32_02);
        fun(SOC2FCT_TBD_u32_03);
        fun(SOC2FCT_TBD_u32_04);
        fun(SOC2FCT_TBD_u32_05);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("msg_ego_nnp", msg_ego_nnp);
        fun("msg_ego_avp", msg_ego_avp);
        fun("SOC2FCT_TBD_u32_01", SOC2FCT_TBD_u32_01);
        fun("SOC2FCT_TBD_u32_02", SOC2FCT_TBD_u32_02);
        fun("SOC2FCT_TBD_u32_03", SOC2FCT_TBD_u32_03);
        fun("SOC2FCT_TBD_u32_04", SOC2FCT_TBD_u32_04);
        fun("SOC2FCT_TBD_u32_05", SOC2FCT_TBD_u32_05);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("msg_ego_nnp", msg_ego_nnp);
        fun("msg_ego_avp", msg_ego_avp);
        fun("SOC2FCT_TBD_u32_01", SOC2FCT_TBD_u32_01);
        fun("SOC2FCT_TBD_u32_02", SOC2FCT_TBD_u32_02);
        fun("SOC2FCT_TBD_u32_03", SOC2FCT_TBD_u32_03);
        fun("SOC2FCT_TBD_u32_04", SOC2FCT_TBD_u32_04);
        fun("SOC2FCT_TBD_u32_05", SOC2FCT_TBD_u32_05);
    }

    bool operator==(const ::hozon::chassis::AlgEgoToMcuFrame& t) const
    {
        return (header == t.header) && (msg_ego_nnp == t.msg_ego_nnp) && (msg_ego_avp == t.msg_ego_avp) && (SOC2FCT_TBD_u32_01 == t.SOC2FCT_TBD_u32_01) && (SOC2FCT_TBD_u32_02 == t.SOC2FCT_TBD_u32_02) && (SOC2FCT_TBD_u32_03 == t.SOC2FCT_TBD_u32_03) && (SOC2FCT_TBD_u32_04 == t.SOC2FCT_TBD_u32_04) && (SOC2FCT_TBD_u32_05 == t.SOC2FCT_TBD_u32_05);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGEGOTOMCUFRAME_H
