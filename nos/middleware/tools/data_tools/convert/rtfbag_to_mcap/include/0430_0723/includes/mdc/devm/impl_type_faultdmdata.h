/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_FAULTDMDATA_H
#define MDC_DEVM_IMPL_TYPE_FAULTDMDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"
#include "mdc/devm/impl_type_didinfolist.h"

namespace mdc {
namespace devm {
struct FaultDmData {
    ::UInt16 eventId;
    ::UInt16 totalSize;
    ::UInt8 dataType;
    ::UInt8 actionTag;
    ::UInt8 actionInfo;
    ::UInt8 freezeFrameTag;
    ::mdc::devm::DidInfoList freezeFrameInfo;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(eventId);
        fun(totalSize);
        fun(dataType);
        fun(actionTag);
        fun(actionInfo);
        fun(freezeFrameTag);
        fun(freezeFrameInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(eventId);
        fun(totalSize);
        fun(dataType);
        fun(actionTag);
        fun(actionInfo);
        fun(freezeFrameTag);
        fun(freezeFrameInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("eventId", eventId);
        fun("totalSize", totalSize);
        fun("dataType", dataType);
        fun("actionTag", actionTag);
        fun("actionInfo", actionInfo);
        fun("freezeFrameTag", freezeFrameTag);
        fun("freezeFrameInfo", freezeFrameInfo);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("eventId", eventId);
        fun("totalSize", totalSize);
        fun("dataType", dataType);
        fun("actionTag", actionTag);
        fun("actionInfo", actionInfo);
        fun("freezeFrameTag", freezeFrameTag);
        fun("freezeFrameInfo", freezeFrameInfo);
    }

    bool operator==(const ::mdc::devm::FaultDmData& t) const
    {
        return (eventId == t.eventId) && (totalSize == t.totalSize) && (dataType == t.dataType) && (actionTag == t.actionTag) && (actionInfo == t.actionInfo) && (freezeFrameTag == t.freezeFrameTag) && (freezeFrameInfo == t.freezeFrameInfo);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_FAULTDMDATA_H
