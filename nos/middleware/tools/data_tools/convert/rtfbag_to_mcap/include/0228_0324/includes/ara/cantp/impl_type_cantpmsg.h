/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CANTP_IMPL_TYPE_CANTPMSG_H
#define ARA_CANTP_IMPL_TYPE_CANTPMSG_H
#include <cfloat>
#include <cmath>
#include "ara/cantp/impl_type_cantpheader.h"
#include "ara/cantp/impl_type_vectoruint8.h"

namespace ara {
namespace cantp {
struct CanTpMsg {
    ::ara::cantp::CanTpHeader header;
    ::ara::cantp::VectorUint8 data;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("data", data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("data", data);
    }

    bool operator==(const ::ara::cantp::CanTpMsg& t) const
    {
        return (header == t.header) && (data == t.data);
    }
};
} // namespace cantp
} // namespace ara


#endif // ARA_CANTP_IMPL_TYPE_CANTPMSG_H
