/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_PLOG_API_PLOG_STAT_STATRETURNCODE_H
#define SRC_PLOG_API_PLOG_STAT_STATRETURNCODE_H

#include <string>

namespace rbs {
namespace plog {
namespace stat {
enum class StatReturnCode : int {
    OK,
    ERROR,
    NOT_VALID,
    ALREADY_BEGIN,
    SIZE_OVERFLOW,
    OUT_OF_RESOURCE,
    NOT_BEGIN,
    OVER_SIZE_NAME,
    DUPLICATE_HANDLE,
    ALREADY_CREATED,
    INFO_WRITER_NUM_UPLIMIT,
    RUN_OUT_DISK_SIZE,
    INVALID_HANDLE_NAME,
    OPEN_ERROR,
    MMAP_ERROR,
    MEM_CPY_ERROR,
    EMPTY_KEY_NAME,
    FALLOCATE_ERROR,
    FLOCK_ERROR
};

/**
 * @brief convert the rtn code to string
 * @param code the code to convert
 * @return a copy of string, which is not a ref because of preventing singleton destruct problem, so this fun cost time
 */
const std::string Code2Str(StatReturnCode code);
}
}
}

#endif // SRC_PLOG_API_PLOG_STAT_STATRETURNCODE_H
