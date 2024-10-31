/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Method result definition
 * Create: 2020-12-24
 */
#ifndef RTF_COM_TYPES_ERROR_CODE_H
#define RTF_COM_TYPES_ERROR_CODE_H
#include <cstdint>
namespace rtf {
namespace com {
enum class ErrorCode : std::uint8_t {
    OK = 0U,
    ERROR,
    TIMEOUT,
    NOTAVAILABLE,
    ERROR_CODE_EX,
    PARAMETER_ERROR
};
} // namespace com
} // namespace rtf
#endif
