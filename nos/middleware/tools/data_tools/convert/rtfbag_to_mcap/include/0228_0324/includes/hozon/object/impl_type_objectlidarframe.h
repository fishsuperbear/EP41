/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_OBJECTLIDARFRAME_H
#define HOZON_OBJECT_IMPL_TYPE_OBJECTLIDARFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/object/impl_type_objectlidarvector3d.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace object {
struct ObjectLidarFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::object::ObjectLidarVector3D object3d;
    ::UInt32 received_ehp_counter;
    ::UInt8 lidarState;
    ::UInt8 isLocationVaild;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(object3d);
        fun(received_ehp_counter);
        fun(lidarState);
        fun(isLocationVaild);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(object3d);
        fun(received_ehp_counter);
        fun(lidarState);
        fun(isLocationVaild);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("object3d", object3d);
        fun("received_ehp_counter", received_ehp_counter);
        fun("lidarState", lidarState);
        fun("isLocationVaild", isLocationVaild);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("object3d", object3d);
        fun("received_ehp_counter", received_ehp_counter);
        fun("lidarState", lidarState);
        fun("isLocationVaild", isLocationVaild);
    }

    bool operator==(const ::hozon::object::ObjectLidarFrame& t) const
    {
        return (header == t.header) && (object3d == t.object3d) && (received_ehp_counter == t.received_ehp_counter) && (lidarState == t.lidarState) && (isLocationVaild == t.isLocationVaild);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_OBJECTLIDARFRAME_H
