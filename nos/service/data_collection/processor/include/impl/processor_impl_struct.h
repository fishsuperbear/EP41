/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: processor_impl_struct.h
 * @Date: 2023/11/14
 * @Author: kun
 * @Desc: --
 */

#pragma once
#include <iostream>
#include <vector>
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace dc {

struct CompressOption {
    std::string model;
    int compressType;
    std::string outputFolderPath;
    std::string outputFileName;
    std::set<std::string> notCompressTypes{};
};

struct FilterOption {
    std::string method;
    std::vector<std::string> whiteTopicVec;
    std::vector<std::string> blackTopicVec;
    std::string outputPath;
};

struct MergeOption {
    std::string method;
    std::vector<std::string> attachmentFilePathVec;
    std::string outputPath;
};

struct SplitOption {
    std::string method;
    std::vector<std::string> attachmentFilePathVec;
    std::string outputPath;
};

struct AddDataOption {
    YAML::Node cmd;
    YAML::Node file;
    std::vector<std::string> calibParamsVec;
};

struct DesenseManagerOption {
    std::string outputFolderPath;
    bool enable;
    uint64_t delayMs;
};

struct AllFileMergeOption {
    std::string outputFolderPath;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon