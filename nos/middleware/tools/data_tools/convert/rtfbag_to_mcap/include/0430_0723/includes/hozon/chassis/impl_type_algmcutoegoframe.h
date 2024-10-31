/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGMCUTOEGOFRAME_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGMCUTOEGOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/chassis/impl_type_algmcuegonnpmsg.h"
#include "hozon/chassis/impl_type_algmcuegoavpmsg.h"
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_algmcuegomemmsg.h"

namespace hozon {
namespace chassis {
struct AlgMcuToEgoFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::chassis::AlgMcuEgoNNPMsg msg_mcu_nnp;
    ::hozon::chassis::AlgMcuEgoAVPMsg msg_mcu_avp;
    ::UInt8 ta_pilot_mode;
    ::UInt32 FCT2SOC_TBD_u32_01;
    ::UInt32 FCT2SOC_TBD_u32_02;
    ::UInt32 FCT2SOC_TBD_u32_03;
    ::UInt32 FCT2SOC_TBD_u32_04;
    ::UInt32 FCT2SOC_TBD_u32_05;
    ::UInt8 drive_mode;
    ::AlgMcuEgoMemMsg msg_mcu_mem;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(msg_mcu_nnp);
        fun(msg_mcu_avp);
        fun(ta_pilot_mode);
        fun(FCT2SOC_TBD_u32_01);
        fun(FCT2SOC_TBD_u32_02);
        fun(FCT2SOC_TBD_u32_03);
        fun(FCT2SOC_TBD_u32_04);
        fun(FCT2SOC_TBD_u32_05);
        fun(drive_mode);
        fun(msg_mcu_mem);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(msg_mcu_nnp);
        fun(msg_mcu_avp);
        fun(ta_pilot_mode);
        fun(FCT2SOC_TBD_u32_01);
        fun(FCT2SOC_TBD_u32_02);
        fun(FCT2SOC_TBD_u32_03);
        fun(FCT2SOC_TBD_u32_04);
        fun(FCT2SOC_TBD_u32_05);
        fun(drive_mode);
        fun(msg_mcu_mem);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("msg_mcu_nnp", msg_mcu_nnp);
        fun("msg_mcu_avp", msg_mcu_avp);
        fun("ta_pilot_mode", ta_pilot_mode);
        fun("FCT2SOC_TBD_u32_01", FCT2SOC_TBD_u32_01);
        fun("FCT2SOC_TBD_u32_02", FCT2SOC_TBD_u32_02);
        fun("FCT2SOC_TBD_u32_03", FCT2SOC_TBD_u32_03);
        fun("FCT2SOC_TBD_u32_04", FCT2SOC_TBD_u32_04);
        fun("FCT2SOC_TBD_u32_05", FCT2SOC_TBD_u32_05);
        fun("drive_mode", drive_mode);
        fun("msg_mcu_mem", msg_mcu_mem);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("msg_mcu_nnp", msg_mcu_nnp);
        fun("msg_mcu_avp", msg_mcu_avp);
        fun("ta_pilot_mode", ta_pilot_mode);
        fun("FCT2SOC_TBD_u32_01", FCT2SOC_TBD_u32_01);
        fun("FCT2SOC_TBD_u32_02", FCT2SOC_TBD_u32_02);
        fun("FCT2SOC_TBD_u32_03", FCT2SOC_TBD_u32_03);
        fun("FCT2SOC_TBD_u32_04", FCT2SOC_TBD_u32_04);
        fun("FCT2SOC_TBD_u32_05", FCT2SOC_TBD_u32_05);
        fun("drive_mode", drive_mode);
        fun("msg_mcu_mem", msg_mcu_mem);
    }

    bool operator==(const ::hozon::chassis::AlgMcuToEgoFrame& t) const
    {
        return (header == t.header) && (msg_mcu_nnp == t.msg_mcu_nnp) && (msg_mcu_avp == t.msg_mcu_avp) && (ta_pilot_mode == t.ta_pilot_mode) && (FCT2SOC_TBD_u32_01 == t.FCT2SOC_TBD_u32_01) && (FCT2SOC_TBD_u32_02 == t.FCT2SOC_TBD_u32_02) && (FCT2SOC_TBD_u32_03 == t.FCT2SOC_TBD_u32_03) && (FCT2SOC_TBD_u32_04 == t.FCT2SOC_TBD_u32_04) && (FCT2SOC_TBD_u32_05 == t.FCT2SOC_TBD_u32_05) && (drive_mode == t.drive_mode) && (msg_mcu_mem == t.msg_mcu_mem);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGMCUTOEGOFRAME_H
