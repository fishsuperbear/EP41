/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DISPLAY_IMPL_TYPE_IMAGEFORMATTYPEE_H
#define ARA_DISPLAY_IMPL_TYPE_IMAGEFORMATTYPEE_H

#include "impl_type_uint8_t.h"
namespace ara {
namespace display {
enum class ImageFormatTypeE : uint8_t
{
    IMAGE_FORMAT_RGB888 = 0U,
    IMAGE_FORMAT_YUV420_NV12 = 1U,
    IMAGE_FORMAT_YUV420_NV21 = 2U
};
} // namespace display
} // namespace ara


#endif // ARA_DISPLAY_IMPL_TYPE_IMAGEFORMATTYPEE_H
