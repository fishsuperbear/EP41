/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: ReturnCode.hpp
 */

#ifndef DDS_CORE_RETURN_CODE_HPP
#define DDS_CORE_RETURN_CODE_HPP

#include <cstdint>
#include <string>

namespace dds {
namespace core {
enum class ReturnCode : std::uint32_t {
    OK,
    ERROR,
    UNSUPPORTED,
    BAD_PARAMETER,
    PRECONDITION_NOT_MET,
    OUT_OF_RESOURCES,
    NOT_ENABLED, /// which is also not created
    ALREADY_DELETED,
    TIMEOUT,
    NO_DATA,
    CREATED_BUT_NOT_ENABLED,
    NULL_REFERENCE,
};

/**
 * @brief a tool fun to convert the code to string
 * @param code the code to convert
 * @return a copy of string, which is not a ref because of preventing singleton destruct problem, so this fun cost time
 */
std::string ReturnCodeToString(ReturnCode code);
}
}

#endif /* DDS_CORE_RETURN_CODE_HPP */

