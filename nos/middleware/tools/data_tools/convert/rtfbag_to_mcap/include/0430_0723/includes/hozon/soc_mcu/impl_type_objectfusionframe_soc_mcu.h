/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_OBJECTFUSIONFRAME_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_OBJECTFUSIONFRAME_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_commonheader_soc_mcu.h"
#include "impl_type_uint32.h"
#include "hozon/soc_mcu/impl_type_objectfusionarray_soc_mcu.h"

namespace hozon {
namespace soc_mcu {
struct ObjectFusionFrame_soc_mcu {
    ::hozon::soc_mcu::CommonHeader_soc_mcu header;
    ::UInt32 locSeq;
    ::hozon::soc_mcu::ObjectFusionArray_soc_mcu object_fusion;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(object_fusion);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(object_fusion);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("object_fusion", object_fusion);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("object_fusion", object_fusion);
    }

    bool operator==(const ::hozon::soc_mcu::ObjectFusionFrame_soc_mcu& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (object_fusion == t.object_fusion);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_OBJECTFUSIONFRAME_SOC_MCU_H
