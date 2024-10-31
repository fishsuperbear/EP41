/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CANFDMSG_IMPL_TYPE_CANFDMSGFRAME_H
#define HOZON_CANFDMSG_IMPL_TYPE_CANFDMSGFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/canfdmsg/impl_type_canfdmsg_vector.h"

namespace hozon {
namespace canfdmsg {
struct CanFdmsgFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::canfdmsg::canfdmsg_vector canfdmsg;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(canfdmsg);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(canfdmsg);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("canfdmsg", canfdmsg);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("canfdmsg", canfdmsg);
    }

    bool operator==(const ::hozon::canfdmsg::CanFdmsgFrame& t) const
    {
        return (header == t.header) && (canfdmsg == t.canfdmsg);
    }
};
} // namespace canfdmsg
} // namespace hozon


#endif // HOZON_CANFDMSG_IMPL_TYPE_CANFDMSGFRAME_H
