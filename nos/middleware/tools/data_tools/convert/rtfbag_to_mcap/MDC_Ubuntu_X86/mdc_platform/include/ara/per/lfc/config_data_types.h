/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: 持久化动态配置数据类型定义
 */

#ifndef ARA_PER_IFC_CONFIG_DATA_TYPES_H
#define ARA_PER_IFC_CONFIG_DATA_TYPES_H

#include "ara/per/per_base_type.h"

namespace ara {
namespace per {
struct RedundancyConfig {
    uint32_t mSize;
    uint32_t nSize;
    uint32_t length;
    ara::core::String algorithmFamily;
    ara::core::String RedundantPath;
    using IsEnumerableTag = void;

    template<typename F>
    void enumerate_internal(F& fun)
    {   
        fun("MinNumValidCopies", mSize, true);
        fun("DatabasesCount", nSize, true);
        fun("AlgorithmFamily", algorithmFamily, true);
        fun("Length", length, true);
        fun("BackupFilePath", RedundantPath, true);
    }
};

struct KeyValueStorageConfig {
    ara::core::String uri;
    ara::core::String redundancy;
    RedundancyConfig bakConfig;
    uint64_t maximumAllowedSize;
    ara::core::String originalFilePath;
    using IsEnumerableTag = void;

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Uri", uri);
        fun("Redundancy", redundancy, true);
        fun("RedundancyConfig", bakConfig, true);
        fun("MaxAllowSize", maximumAllowedSize, true);
        fun("OriginalFilePath", originalFilePath, true);
    }
};

struct FileStorageConfig {
    ara::core::String uri;
    ara::core::Vector<ara::core::String> persistencyFiles;
    ara::core::String redundancy;
    RedundancyConfig bakConfig;
    uint64_t maximumAllowedSize;
    ara::core::String originalFilePath;
    using IsEnumerableTag = void;

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("RedundancyConfig", bakConfig, true);
        fun("MaxAllowSize", maximumAllowedSize, true);
        fun("OriginalFilePath", originalFilePath, true);
        fun("Uri", uri);
        fun("Redundancy", redundancy, true);
    }
};
}  // namespace per
}  // namespace ara
#endif  // ARA_PER_IFC_CONFIG_DATA_TYPES_H
