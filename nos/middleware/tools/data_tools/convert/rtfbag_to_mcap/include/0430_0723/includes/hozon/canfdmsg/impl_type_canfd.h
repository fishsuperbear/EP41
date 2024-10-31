/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CANFDMSG_IMPL_TYPE_CANFD_H
#define HOZON_CANFDMSG_IMPL_TYPE_CANFD_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "hozon/canfdmsg/impl_type_canmsg_array.h"

namespace hozon {
namespace canfdmsg {
struct canfd {
    ::uint32_t canid;
    ::hozon::canfdmsg::canmsg_array msg;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(canid);
        fun(msg);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(canid);
        fun(msg);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("canid", canid);
        fun("msg", msg);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("canid", canid);
        fun("msg", msg);
    }

    bool operator==(const ::hozon::canfdmsg::canfd& t) const
    {
        return (canid == t.canid) && (msg == t.msg);
    }
};
} // namespace canfdmsg
} // namespace hozon


#endif // HOZON_CANFDMSG_IMPL_TYPE_CANFD_H
