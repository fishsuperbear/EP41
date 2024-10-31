/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: 持久化功能配置初始化接口
 */

#ifndef ARA_PER_LFC_INITIALIZATION_H
#define ARA_PER_LFC_INITIALIZATION_H

#include "ara/per/lfc/config_data_types.h"

namespace ara {
namespace per {
namespace lfc {
ara::core::Result<void> Initialize();

ara::core::Result<void> Initialize(const ara::core::String& name, const KeyValueStorageConfig& config, const bool& doDeploy = true);

ara::core::Result<void> Initialize(const ara::core::String& name, const FileStorageConfig& config, const bool& doDeploy = true);
}  // namespace ifc
}  // namespace per
}  // namespace ara
#endif  // ARA_PER_LFC_INITIALIZATION_H

