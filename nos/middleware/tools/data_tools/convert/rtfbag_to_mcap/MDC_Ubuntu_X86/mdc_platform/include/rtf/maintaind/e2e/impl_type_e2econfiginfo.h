/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_IMPL_TYPE_E2ECONFIGINFO_H
#define RTF_MAINTAIND_IMPL_TYPE_E2ECONFIGINFO_H
#include "ara/core/map.h"
#include "rtf/maintaind/e2e/impl_type_e2eprofile.h"
#include "rtf/basetype/impl_type_vectoruint32.h"
#include "rtf/stdtype/impl_type_uint32_t.h"
#include "rtf/stdtype/impl_type_uint16_t.h"
#include "rtf/maintaind/e2e/impl_type_e2edataidmode.h"
#include "rtf/stdtype/impl_type_uint8_t.h"

namespace rtf {
namespace maintaind {
struct E2EConfigInfo {
    ::rtf::maintaind::E2EProfile profile = E2EProfile::UNDEFINE;
    ::rtf::basetype::VectorUint32 dataIdList;
    bool isDisableE2ECheck = false;
    bool isEnableCRCHardware = false;
    ::rtf::stdtype::uint32_t minDataLength = 0U;
    ::rtf::stdtype::uint32_t maxDataLength = 0U;
    ::rtf::stdtype::uint16_t dataLength = 0U;
    ::rtf::stdtype::uint32_t maxDeltaCounter = 0U;
    ::rtf::maintaind::E2EDataIDMode dataIdMode = E2EDataIDMode::E2E_DATAID_NIBBLE;
    ::rtf::stdtype::uint8_t windowSizeValid = 0U;
    ::rtf::stdtype::uint8_t minOkStateInit = 0U;
    ::rtf::stdtype::uint8_t maxErrorStateInit = 0U;
    ::rtf::stdtype::uint8_t minOkStateValid = 0U;
    ::rtf::stdtype::uint8_t maxErrorStateValid = 0U;
    ::rtf::stdtype::uint8_t minOkStateInvalid = 0U;
    ::rtf::stdtype::uint8_t maxErrorStateInvalid = 0U;
    ::rtf::stdtype::uint8_t windowSizeInit = 0U;
    ::rtf::stdtype::uint8_t windowSizeInvalid = 0U;
    bool clearToInvalid = false;

    ::rtf::stdtype::uint32_t offset = 0U;
    static bool IsPlane() noexcept
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun) noexcept
    {
        fun(profile);
        fun(dataIdList);
        fun(isDisableE2ECheck);
        fun(isEnableCRCHardware);
        fun(minDataLength);
        fun(maxDataLength);
        fun(dataLength);
        fun(maxDeltaCounter);
        fun(dataIdMode);
        fun(windowSizeValid);
        fun(minOkStateInit);
        fun(maxErrorStateInit);
        fun(minOkStateValid);
        fun(maxErrorStateValid);
        fun(minOkStateInvalid);
        fun(maxErrorStateInvalid);
        fun(windowSizeInit);
        fun(windowSizeInvalid);
        fun(clearToInvalid);
        fun(offset);
    }

    bool operator == (const ::rtf::maintaind::E2EConfigInfo& t) const noexcept
    {
        return (profile == t.profile) && (dataIdList == t.dataIdList) && (isDisableE2ECheck == t.isDisableE2ECheck) &&
        (isEnableCRCHardware == t.isEnableCRCHardware) && (minDataLength == t.minDataLength) &&
        (maxDataLength == t.maxDataLength) && (dataLength == t.dataLength) && (maxDeltaCounter == t.maxDeltaCounter) &&
        (dataIdMode == t.dataIdMode) && (windowSizeValid == t.windowSizeValid) &&
        (minOkStateInit == t.minOkStateInit) && (maxErrorStateInit == t.maxErrorStateInit) &&
        (minOkStateValid == t.minOkStateValid) && (maxErrorStateValid == t.maxErrorStateValid) &&
        (minOkStateInvalid == t.minOkStateInvalid) && (maxErrorStateInvalid == t.maxErrorStateInvalid) &&
        (windowSizeInit == t.windowSizeInit) && (windowSizeInvalid == t.windowSizeInvalid) &&
        (clearToInvalid == t.clearToInvalid) && (offset == t.offset);
    }
};
} // namespace maintaind
} // namespace rtf


#endif // RTF_MAINTAIND_IMPL_TYPE_E2ECONFIGINFO_H
