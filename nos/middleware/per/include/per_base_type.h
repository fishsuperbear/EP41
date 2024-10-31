/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 基本数据类型和冗余配置定义
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_PER_BASE_TYPE_H_
#define MIDDLEWARE_PER_INCLUDE_PER_BASE_TYPE_H_
#include <string>
#include <vector>
// #include "core/string.h"
// #include "core/map.h"
// #include "core/vector.h"
// #include "core/span.h"
// #include "core/result.h"

namespace hozon {
namespace netaos {
namespace per {
using char_t = char;
using uchar_t = unsigned char;
using float32_t = float;
using float64_t = double;
using Container = std::vector<std::uint8_t>;
struct RedundancyConfig {
    uint8_t redundant_count = 0;           // 备份数
    std::string redundant_dirpath = "./";  // 备份文件夹路径
    bool auto_recover = false;             // 自动恢复开关
};
struct StorageConfig {
    std::string original_file_path = "./";  // 初始化文件路径
    std::string serialize_format = "";      // 文件格式
    RedundancyConfig redundancy_config;     // 冗余配置
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_PER_BASE_TYPE_H_
