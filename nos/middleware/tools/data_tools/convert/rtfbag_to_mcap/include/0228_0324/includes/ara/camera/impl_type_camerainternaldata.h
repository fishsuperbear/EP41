/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CAMERA_IMPL_TYPE_CAMERAINTERNALDATA_H
#define ARA_CAMERA_IMPL_TYPE_CAMERAINTERNALDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_uint8vector.h"
#include "ara/camera/impl_type_moduleinfo.h"

namespace ara {
namespace camera {
struct CameraInternalData {
    ::Boolean isValid;
    ::Uint8Vector data;
    ::ara::camera::ModuleInfo moduleInfo;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isValid);
        fun(data);
        fun(moduleInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isValid);
        fun(data);
        fun(moduleInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("isValid", isValid);
        fun("data", data);
        fun("moduleInfo", moduleInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("isValid", isValid);
        fun("data", data);
        fun("moduleInfo", moduleInfo);
    }

    bool operator==(const ::ara::camera::CameraInternalData& t) const
    {
        return (isValid == t.isValid) && (data == t.data) && (moduleInfo == t.moduleInfo);
    }
};
} // namespace camera
} // namespace ara


#endif // ARA_CAMERA_IMPL_TYPE_CAMERAINTERNALDATA_H
