/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_E2ESTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_E2ESTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_INTF_E2EStatus {
    ::UInt8 INTF_E2E_FD3_0x137;
    ::UInt8 INTF_E2E_FD3_0x138;
    ::UInt8 INTF_E2E_FD3_0x139;
    ::UInt8 INTF_E2E_FD3_0x13B;
    ::UInt8 INTF_E2E_FD3_0x13C;
    ::UInt8 INTF_E2E_FD3_0x13D;
    ::UInt8 INTF_E2E_FD3_0x13E;
    ::UInt8 INTF_E2E_FD3_0xC4;
    ::UInt8 INTF_E2E_FD3_0x110;
    ::UInt8 INTF_E2E_FD3_0x114;
    ::UInt8 INTF_E2E_FD3_0xF3;
    ::UInt8 INTF_E2E_FD3_0xB1;
    ::UInt8 INTF_E2E_FD3_0xB2;
    ::UInt8 INTF_E2E_FD3_0xAB;
    ::UInt8 INTF_E2E_FD3_0xC0;
    ::UInt8 INTF_E2E_FD3_0xC5;
    ::UInt8 INTF_E2E_FD3_0xC7;
    ::UInt8 INTF_E2E_FD3_0xE5;
    ::UInt8 INTF_E2E_FD3_0x121;
    ::UInt8 INTF_E2E_FD3_0x129;
    ::UInt8 INTF_E2E_FD3_0x108;
    ::UInt8 INTF_E2E_FD3_0x1B6;
    ::UInt8 INTF_E2E_FD3_0xE3;
    ::UInt8 INTF_E2E_FD3_0x12D;
    ::UInt8 INTF_E2E_FD6_0x110;
    ::UInt8 INTF_E2E_FD8_0x137;
    ::UInt8 INTF_E2E_FD8_0x138;
    ::UInt8 INTF_E2E_FD8_0x139;
    ::UInt8 INTF_E2E_FD8_0x13B;
    ::UInt8 INTF_E2E_FD8_0x13C;
    ::UInt8 INTF_E2E_FD8_0x13D;
    ::UInt8 INTF_E2E_FD8_0x13E;
    ::UInt8 INTF_E2E_FD8_0x2FE;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_E2E_FD3_0x137);
        fun(INTF_E2E_FD3_0x138);
        fun(INTF_E2E_FD3_0x139);
        fun(INTF_E2E_FD3_0x13B);
        fun(INTF_E2E_FD3_0x13C);
        fun(INTF_E2E_FD3_0x13D);
        fun(INTF_E2E_FD3_0x13E);
        fun(INTF_E2E_FD3_0xC4);
        fun(INTF_E2E_FD3_0x110);
        fun(INTF_E2E_FD3_0x114);
        fun(INTF_E2E_FD3_0xF3);
        fun(INTF_E2E_FD3_0xB1);
        fun(INTF_E2E_FD3_0xB2);
        fun(INTF_E2E_FD3_0xAB);
        fun(INTF_E2E_FD3_0xC0);
        fun(INTF_E2E_FD3_0xC5);
        fun(INTF_E2E_FD3_0xC7);
        fun(INTF_E2E_FD3_0xE5);
        fun(INTF_E2E_FD3_0x121);
        fun(INTF_E2E_FD3_0x129);
        fun(INTF_E2E_FD3_0x108);
        fun(INTF_E2E_FD3_0x1B6);
        fun(INTF_E2E_FD3_0xE3);
        fun(INTF_E2E_FD3_0x12D);
        fun(INTF_E2E_FD6_0x110);
        fun(INTF_E2E_FD8_0x137);
        fun(INTF_E2E_FD8_0x138);
        fun(INTF_E2E_FD8_0x139);
        fun(INTF_E2E_FD8_0x13B);
        fun(INTF_E2E_FD8_0x13C);
        fun(INTF_E2E_FD8_0x13D);
        fun(INTF_E2E_FD8_0x13E);
        fun(INTF_E2E_FD8_0x2FE);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_E2E_FD3_0x137);
        fun(INTF_E2E_FD3_0x138);
        fun(INTF_E2E_FD3_0x139);
        fun(INTF_E2E_FD3_0x13B);
        fun(INTF_E2E_FD3_0x13C);
        fun(INTF_E2E_FD3_0x13D);
        fun(INTF_E2E_FD3_0x13E);
        fun(INTF_E2E_FD3_0xC4);
        fun(INTF_E2E_FD3_0x110);
        fun(INTF_E2E_FD3_0x114);
        fun(INTF_E2E_FD3_0xF3);
        fun(INTF_E2E_FD3_0xB1);
        fun(INTF_E2E_FD3_0xB2);
        fun(INTF_E2E_FD3_0xAB);
        fun(INTF_E2E_FD3_0xC0);
        fun(INTF_E2E_FD3_0xC5);
        fun(INTF_E2E_FD3_0xC7);
        fun(INTF_E2E_FD3_0xE5);
        fun(INTF_E2E_FD3_0x121);
        fun(INTF_E2E_FD3_0x129);
        fun(INTF_E2E_FD3_0x108);
        fun(INTF_E2E_FD3_0x1B6);
        fun(INTF_E2E_FD3_0xE3);
        fun(INTF_E2E_FD3_0x12D);
        fun(INTF_E2E_FD6_0x110);
        fun(INTF_E2E_FD8_0x137);
        fun(INTF_E2E_FD8_0x138);
        fun(INTF_E2E_FD8_0x139);
        fun(INTF_E2E_FD8_0x13B);
        fun(INTF_E2E_FD8_0x13C);
        fun(INTF_E2E_FD8_0x13D);
        fun(INTF_E2E_FD8_0x13E);
        fun(INTF_E2E_FD8_0x2FE);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_E2E_FD3_0x137", INTF_E2E_FD3_0x137);
        fun("INTF_E2E_FD3_0x138", INTF_E2E_FD3_0x138);
        fun("INTF_E2E_FD3_0x139", INTF_E2E_FD3_0x139);
        fun("INTF_E2E_FD3_0x13B", INTF_E2E_FD3_0x13B);
        fun("INTF_E2E_FD3_0x13C", INTF_E2E_FD3_0x13C);
        fun("INTF_E2E_FD3_0x13D", INTF_E2E_FD3_0x13D);
        fun("INTF_E2E_FD3_0x13E", INTF_E2E_FD3_0x13E);
        fun("INTF_E2E_FD3_0xC4", INTF_E2E_FD3_0xC4);
        fun("INTF_E2E_FD3_0x110", INTF_E2E_FD3_0x110);
        fun("INTF_E2E_FD3_0x114", INTF_E2E_FD3_0x114);
        fun("INTF_E2E_FD3_0xF3", INTF_E2E_FD3_0xF3);
        fun("INTF_E2E_FD3_0xB1", INTF_E2E_FD3_0xB1);
        fun("INTF_E2E_FD3_0xB2", INTF_E2E_FD3_0xB2);
        fun("INTF_E2E_FD3_0xAB", INTF_E2E_FD3_0xAB);
        fun("INTF_E2E_FD3_0xC0", INTF_E2E_FD3_0xC0);
        fun("INTF_E2E_FD3_0xC5", INTF_E2E_FD3_0xC5);
        fun("INTF_E2E_FD3_0xC7", INTF_E2E_FD3_0xC7);
        fun("INTF_E2E_FD3_0xE5", INTF_E2E_FD3_0xE5);
        fun("INTF_E2E_FD3_0x121", INTF_E2E_FD3_0x121);
        fun("INTF_E2E_FD3_0x129", INTF_E2E_FD3_0x129);
        fun("INTF_E2E_FD3_0x108", INTF_E2E_FD3_0x108);
        fun("INTF_E2E_FD3_0x1B6", INTF_E2E_FD3_0x1B6);
        fun("INTF_E2E_FD3_0xE3", INTF_E2E_FD3_0xE3);
        fun("INTF_E2E_FD3_0x12D", INTF_E2E_FD3_0x12D);
        fun("INTF_E2E_FD6_0x110", INTF_E2E_FD6_0x110);
        fun("INTF_E2E_FD8_0x137", INTF_E2E_FD8_0x137);
        fun("INTF_E2E_FD8_0x138", INTF_E2E_FD8_0x138);
        fun("INTF_E2E_FD8_0x139", INTF_E2E_FD8_0x139);
        fun("INTF_E2E_FD8_0x13B", INTF_E2E_FD8_0x13B);
        fun("INTF_E2E_FD8_0x13C", INTF_E2E_FD8_0x13C);
        fun("INTF_E2E_FD8_0x13D", INTF_E2E_FD8_0x13D);
        fun("INTF_E2E_FD8_0x13E", INTF_E2E_FD8_0x13E);
        fun("INTF_E2E_FD8_0x2FE", INTF_E2E_FD8_0x2FE);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_E2E_FD3_0x137", INTF_E2E_FD3_0x137);
        fun("INTF_E2E_FD3_0x138", INTF_E2E_FD3_0x138);
        fun("INTF_E2E_FD3_0x139", INTF_E2E_FD3_0x139);
        fun("INTF_E2E_FD3_0x13B", INTF_E2E_FD3_0x13B);
        fun("INTF_E2E_FD3_0x13C", INTF_E2E_FD3_0x13C);
        fun("INTF_E2E_FD3_0x13D", INTF_E2E_FD3_0x13D);
        fun("INTF_E2E_FD3_0x13E", INTF_E2E_FD3_0x13E);
        fun("INTF_E2E_FD3_0xC4", INTF_E2E_FD3_0xC4);
        fun("INTF_E2E_FD3_0x110", INTF_E2E_FD3_0x110);
        fun("INTF_E2E_FD3_0x114", INTF_E2E_FD3_0x114);
        fun("INTF_E2E_FD3_0xF3", INTF_E2E_FD3_0xF3);
        fun("INTF_E2E_FD3_0xB1", INTF_E2E_FD3_0xB1);
        fun("INTF_E2E_FD3_0xB2", INTF_E2E_FD3_0xB2);
        fun("INTF_E2E_FD3_0xAB", INTF_E2E_FD3_0xAB);
        fun("INTF_E2E_FD3_0xC0", INTF_E2E_FD3_0xC0);
        fun("INTF_E2E_FD3_0xC5", INTF_E2E_FD3_0xC5);
        fun("INTF_E2E_FD3_0xC7", INTF_E2E_FD3_0xC7);
        fun("INTF_E2E_FD3_0xE5", INTF_E2E_FD3_0xE5);
        fun("INTF_E2E_FD3_0x121", INTF_E2E_FD3_0x121);
        fun("INTF_E2E_FD3_0x129", INTF_E2E_FD3_0x129);
        fun("INTF_E2E_FD3_0x108", INTF_E2E_FD3_0x108);
        fun("INTF_E2E_FD3_0x1B6", INTF_E2E_FD3_0x1B6);
        fun("INTF_E2E_FD3_0xE3", INTF_E2E_FD3_0xE3);
        fun("INTF_E2E_FD3_0x12D", INTF_E2E_FD3_0x12D);
        fun("INTF_E2E_FD6_0x110", INTF_E2E_FD6_0x110);
        fun("INTF_E2E_FD8_0x137", INTF_E2E_FD8_0x137);
        fun("INTF_E2E_FD8_0x138", INTF_E2E_FD8_0x138);
        fun("INTF_E2E_FD8_0x139", INTF_E2E_FD8_0x139);
        fun("INTF_E2E_FD8_0x13B", INTF_E2E_FD8_0x13B);
        fun("INTF_E2E_FD8_0x13C", INTF_E2E_FD8_0x13C);
        fun("INTF_E2E_FD8_0x13D", INTF_E2E_FD8_0x13D);
        fun("INTF_E2E_FD8_0x13E", INTF_E2E_FD8_0x13E);
        fun("INTF_E2E_FD8_0x2FE", INTF_E2E_FD8_0x2FE);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_INTF_E2EStatus& t) const
    {
        return (INTF_E2E_FD3_0x137 == t.INTF_E2E_FD3_0x137) && (INTF_E2E_FD3_0x138 == t.INTF_E2E_FD3_0x138) && (INTF_E2E_FD3_0x139 == t.INTF_E2E_FD3_0x139) && (INTF_E2E_FD3_0x13B == t.INTF_E2E_FD3_0x13B) && (INTF_E2E_FD3_0x13C == t.INTF_E2E_FD3_0x13C) && (INTF_E2E_FD3_0x13D == t.INTF_E2E_FD3_0x13D) && (INTF_E2E_FD3_0x13E == t.INTF_E2E_FD3_0x13E) && (INTF_E2E_FD3_0xC4 == t.INTF_E2E_FD3_0xC4) && (INTF_E2E_FD3_0x110 == t.INTF_E2E_FD3_0x110) && (INTF_E2E_FD3_0x114 == t.INTF_E2E_FD3_0x114) && (INTF_E2E_FD3_0xF3 == t.INTF_E2E_FD3_0xF3) && (INTF_E2E_FD3_0xB1 == t.INTF_E2E_FD3_0xB1) && (INTF_E2E_FD3_0xB2 == t.INTF_E2E_FD3_0xB2) && (INTF_E2E_FD3_0xAB == t.INTF_E2E_FD3_0xAB) && (INTF_E2E_FD3_0xC0 == t.INTF_E2E_FD3_0xC0) && (INTF_E2E_FD3_0xC5 == t.INTF_E2E_FD3_0xC5) && (INTF_E2E_FD3_0xC7 == t.INTF_E2E_FD3_0xC7) && (INTF_E2E_FD3_0xE5 == t.INTF_E2E_FD3_0xE5) && (INTF_E2E_FD3_0x121 == t.INTF_E2E_FD3_0x121) && (INTF_E2E_FD3_0x129 == t.INTF_E2E_FD3_0x129) && (INTF_E2E_FD3_0x108 == t.INTF_E2E_FD3_0x108) && (INTF_E2E_FD3_0x1B6 == t.INTF_E2E_FD3_0x1B6) && (INTF_E2E_FD3_0xE3 == t.INTF_E2E_FD3_0xE3) && (INTF_E2E_FD3_0x12D == t.INTF_E2E_FD3_0x12D) && (INTF_E2E_FD6_0x110 == t.INTF_E2E_FD6_0x110) && (INTF_E2E_FD8_0x137 == t.INTF_E2E_FD8_0x137) && (INTF_E2E_FD8_0x138 == t.INTF_E2E_FD8_0x138) && (INTF_E2E_FD8_0x139 == t.INTF_E2E_FD8_0x139) && (INTF_E2E_FD8_0x13B == t.INTF_E2E_FD8_0x13B) && (INTF_E2E_FD8_0x13C == t.INTF_E2E_FD8_0x13C) && (INTF_E2E_FD8_0x13D == t.INTF_E2E_FD8_0x13D) && (INTF_E2E_FD8_0x13E == t.INTF_E2E_FD8_0x13E) && (INTF_E2E_FD8_0x2FE == t.INTF_E2E_FD8_0x2FE);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_E2ESTATUS_H
