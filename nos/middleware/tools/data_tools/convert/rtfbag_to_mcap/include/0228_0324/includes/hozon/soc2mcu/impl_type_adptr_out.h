/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_ADPTR_OUT_H
#define HOZON_SOC2MCU_IMPL_TYPE_ADPTR_OUT_H
#include <cfloat>
#include <cmath>
#include "hozon/soc2mcu/impl_type_adas_msg_0x8e.h"
#include "hozon/soc2mcu/impl_type_adas_msg_0x8f.h"
#include "hozon/soc2mcu/impl_type_adas_msg_0x136.h"
#include "hozon/soc2mcu/impl_type_adas_msg_0x193.h"

namespace hozon {
namespace soc2mcu {
struct Adptr_Out {
    ::hozon::soc2mcu::Adas_MSG_0x8E Adptr_Out_Msg_0x8E;
    ::hozon::soc2mcu::Adas_MSG_0x8F Adptr_Out_Msg_0x8F;
    ::hozon::soc2mcu::Adas_MSG_0x136 Adptr_Out_Msg_0x136;
    ::hozon::soc2mcu::Adas_MSG_0x193 Adptr_Out_Msg_0x193;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Adptr_Out_Msg_0x8E);
        fun(Adptr_Out_Msg_0x8F);
        fun(Adptr_Out_Msg_0x136);
        fun(Adptr_Out_Msg_0x193);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Adptr_Out_Msg_0x8E);
        fun(Adptr_Out_Msg_0x8F);
        fun(Adptr_Out_Msg_0x136);
        fun(Adptr_Out_Msg_0x193);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Adptr_Out_Msg_0x8E", Adptr_Out_Msg_0x8E);
        fun("Adptr_Out_Msg_0x8F", Adptr_Out_Msg_0x8F);
        fun("Adptr_Out_Msg_0x136", Adptr_Out_Msg_0x136);
        fun("Adptr_Out_Msg_0x193", Adptr_Out_Msg_0x193);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Adptr_Out_Msg_0x8E", Adptr_Out_Msg_0x8E);
        fun("Adptr_Out_Msg_0x8F", Adptr_Out_Msg_0x8F);
        fun("Adptr_Out_Msg_0x136", Adptr_Out_Msg_0x136);
        fun("Adptr_Out_Msg_0x193", Adptr_Out_Msg_0x193);
    }

    bool operator==(const ::hozon::soc2mcu::Adptr_Out& t) const
    {
        return (Adptr_Out_Msg_0x8E == t.Adptr_Out_Msg_0x8E) && (Adptr_Out_Msg_0x8F == t.Adptr_Out_Msg_0x8F) && (Adptr_Out_Msg_0x136 == t.Adptr_Out_Msg_0x136) && (Adptr_Out_Msg_0x193 == t.Adptr_Out_Msg_0x193);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_ADPTR_OUT_H
