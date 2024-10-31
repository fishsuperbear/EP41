/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DISPLAY_IMPL_TYPE_ROTATEMODEE_H
#define ARA_DISPLAY_IMPL_TYPE_ROTATEMODEE_H

#include "impl_type_uint8_t.h"
namespace ara {
namespace display {
enum class RotateModeE : uint8_t
{
    DO_NOT_ROTATE = 0U,
    ROTATE_MODE_90_CLOCKWISE = 1U,
    ROTATE_MODE_180_CLOCKWISE = 2U,
    ROTATE_MODE_270_CLOCKWISE = 3U
};
} // namespace display
} // namespace ara


#endif // ARA_DISPLAY_IMPL_TYPE_ROTATEMODEE_H
