/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ALGPERCEPTIONINFOFRAME_H
#define IMPL_TYPE_ALGPERCEPTIONINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_algtsrmsg.h"

struct AlgPerceptionInfoFrame {
    ::hozon::common::CommonHeader header;
    ::AlgTsrMsg tsr_info;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(tsr_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(tsr_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("tsr_info", tsr_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("tsr_info", tsr_info);
    }

    bool operator==(const ::AlgPerceptionInfoFrame& t) const
    {
        return (header == t.header) && (tsr_info == t.tsr_info);
    }
};


#endif // IMPL_TYPE_ALGPERCEPTIONINFOFRAME_H
