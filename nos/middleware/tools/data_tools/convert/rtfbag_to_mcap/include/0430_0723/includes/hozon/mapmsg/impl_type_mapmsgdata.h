/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_MAPMSG_IMPL_TYPE_MAPMSGDATA_H
#define HOZON_MAPMSG_IMPL_TYPE_MAPMSGDATA_H
#include <cfloat>
#include <cmath>
#include "hozon/mapmsg/impl_type_stringvector.h"
#include "impl_type_int32.h"

namespace hozon {
namespace mapmsg {
struct MapMsgData {
    ::hozon::mapmsg::stringVector map_msg;
    ::Int32 send_counter;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(map_msg);
        fun(send_counter);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(map_msg);
        fun(send_counter);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("map_msg", map_msg);
        fun("send_counter", send_counter);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("map_msg", map_msg);
        fun("send_counter", send_counter);
    }

    bool operator==(const ::hozon::mapmsg::MapMsgData& t) const
    {
        return (map_msg == t.map_msg) && (send_counter == t.send_counter);
    }
};
} // namespace mapmsg
} // namespace hozon


#endif // HOZON_MAPMSG_IMPL_TYPE_MAPMSGDATA_H
