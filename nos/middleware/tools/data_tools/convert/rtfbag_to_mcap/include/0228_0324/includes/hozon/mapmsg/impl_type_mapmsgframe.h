/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_MAPMSG_IMPL_TYPE_MAPMSGFRAME_H
#define HOZON_MAPMSG_IMPL_TYPE_MAPMSGFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/mapmsg/impl_type_msgvector.h"

namespace hozon {
namespace mapmsg {
struct MapMsgFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::mapmsg::msgVector msg_vec;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(msg_vec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(msg_vec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("msg_vec", msg_vec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("msg_vec", msg_vec);
    }

    bool operator==(const ::hozon::mapmsg::MapMsgFrame& t) const
    {
        return (header == t.header) && (msg_vec == t.msg_vec);
    }
};
} // namespace mapmsg
} // namespace hozon


#endif // HOZON_MAPMSG_IMPL_TYPE_MAPMSGFRAME_H
