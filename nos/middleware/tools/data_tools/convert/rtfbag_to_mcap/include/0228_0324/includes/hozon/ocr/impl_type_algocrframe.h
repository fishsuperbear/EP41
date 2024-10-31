/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OCR_IMPL_TYPE_ALGOCRFRAME_H
#define HOZON_OCR_IMPL_TYPE_ALGOCRFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/ocr/impl_type_arrayocr.h"

namespace hozon {
namespace ocr {
struct AlgOcrFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::ocr::arrayocr ocrarray;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(ocrarray);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(ocrarray);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("ocrarray", ocrarray);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("ocrarray", ocrarray);
    }

    bool operator==(const ::hozon::ocr::AlgOcrFrame& t) const
    {
        return (header == t.header) && (ocrarray == t.ocrarray);
    }
};
} // namespace ocr
} // namespace hozon


#endif // HOZON_OCR_IMPL_TYPE_ALGOCRFRAME_H
