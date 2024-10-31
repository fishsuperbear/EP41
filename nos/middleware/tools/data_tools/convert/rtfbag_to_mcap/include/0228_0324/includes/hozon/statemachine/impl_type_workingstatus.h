/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_WORKINGSTATUS_H
#define HOZON_STATEMACHINE_IMPL_TYPE_WORKINGSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace statemachine {
struct WorkingStatus {
    ::UInt8 processing_status;
    ::UInt8 error_code;
    ::UInt8 perception_warninginfo;
    ::UInt8 perception_ADCS4_Tex;
    ::UInt8 perception_ADCS4_PA_failinfo;
    ::UInt8 TBA_Distance;
    ::UInt8 TBA;
    ::UInt8 TBA_text;
    ::uint8_t reserved2;
    ::uint8_t reserved3;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(processing_status);
        fun(error_code);
        fun(perception_warninginfo);
        fun(perception_ADCS4_Tex);
        fun(perception_ADCS4_PA_failinfo);
        fun(TBA_Distance);
        fun(TBA);
        fun(TBA_text);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(processing_status);
        fun(error_code);
        fun(perception_warninginfo);
        fun(perception_ADCS4_Tex);
        fun(perception_ADCS4_PA_failinfo);
        fun(TBA_Distance);
        fun(TBA);
        fun(TBA_text);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("processing_status", processing_status);
        fun("error_code", error_code);
        fun("perception_warninginfo", perception_warninginfo);
        fun("perception_ADCS4_Tex", perception_ADCS4_Tex);
        fun("perception_ADCS4_PA_failinfo", perception_ADCS4_PA_failinfo);
        fun("TBA_Distance", TBA_Distance);
        fun("TBA", TBA);
        fun("TBA_text", TBA_text);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("processing_status", processing_status);
        fun("error_code", error_code);
        fun("perception_warninginfo", perception_warninginfo);
        fun("perception_ADCS4_Tex", perception_ADCS4_Tex);
        fun("perception_ADCS4_PA_failinfo", perception_ADCS4_PA_failinfo);
        fun("TBA_Distance", TBA_Distance);
        fun("TBA", TBA);
        fun("TBA_text", TBA_text);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    bool operator==(const ::hozon::statemachine::WorkingStatus& t) const
    {
        return (processing_status == t.processing_status) && (error_code == t.error_code) && (perception_warninginfo == t.perception_warninginfo) && (perception_ADCS4_Tex == t.perception_ADCS4_Tex) && (perception_ADCS4_PA_failinfo == t.perception_ADCS4_PA_failinfo) && (TBA_Distance == t.TBA_Distance) && (TBA == t.TBA) && (TBA_text == t.TBA_text) && (reserved2 == t.reserved2) && (reserved3 == t.reserved3);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_WORKINGSTATUS_H
