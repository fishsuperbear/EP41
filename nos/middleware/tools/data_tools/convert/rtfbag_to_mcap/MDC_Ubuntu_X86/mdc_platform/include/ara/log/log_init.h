/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Log init head file
 * Create: 2021-7-12
 */
#ifndef ARR_LOG_INIT_H
#define ARR_LOG_INIT_H

#include "ara/core/result.h"

namespace ara {
namespace log {
ara::core::Result<void> Initialize();
ara::core::Result<void> Deinitialize() noexcept;
} // namespace log
} // namespace ara
#endif  // ARR_LOG_INIT_H
