// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_compression_base.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 
/// @date 2023-11-15

#include "log_collector/include/log_file_compression_base.h"

#include <unistd.h>
#include <iostream>

#include "spdlog/details/os.h"

namespace hozon {
namespace netaos {
namespace logcollector {
 
LogFileCompressionBase::LogFileCompressionBase() {
}

LogFileCompressionBase::~LogFileCompressionBase() {
}

bool LogFileCompressionBase::RenameFile(const spdlog::filename_t &src_filename,
            const spdlog::filename_t &target_filename) {
    (void)spdlog::details::os::remove(target_filename);
    return spdlog::details::os::rename(src_filename, target_filename) == 0;
}

std::string LogFileCompressionBase::GetAbsolutePath(const std::string& relative_path) {
    // 如果输入的是绝对路径，则直接返回
    if (!relative_path.empty() && relative_path.front() == '/') {
        return relative_path;
    }

    // 初始为空
    std::string absolutePath = "";  
    // 获取当前工作目录的绝对路径
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        absolutePath = cwd;
    } else {
        std::cerr << "Failed to get current working directory." << std::endl;
        // 获取当前工作目录失败，返回空字符串
        return "";
    }

    // 将相对路径附加到绝对路径后面
    if (!relative_path.empty() && relative_path != "./") {
        absolutePath += "/" + relative_path;
    }

    if (absolutePath.back() != '/') {
        // 如果绝对路径最后没有斜杠，则添加斜杠
        absolutePath += "/";
    }

    return absolutePath;
}

} // namespace logcollector
} // namespace netaos
} // namespace hozon
