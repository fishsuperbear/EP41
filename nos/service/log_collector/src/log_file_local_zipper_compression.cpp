// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_local_zipper_compression.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 
/// @date 2023-11-15

#include "log_collector/include/log_file_local_zipper_compression.h"

#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/fmt/fmt.h>

#include <zipper.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include "spdlog/common.h"
#include "spdlog/details/file_helper.h"
#include "spdlog/details/os.h"

namespace hozon {
namespace netaos {
namespace logcollector {
 
LogFileLocalZipperCompression::LogFileLocalZipperCompression() :
    LogFileCompressionBase() {
}

LogFileLocalZipperCompression::~LogFileLocalZipperCompression() {
}

bool LogFileLocalZipperCompression::DO(const std::string &appid,
            const std::string &file_path, const std::string &file_name, std::string &zip_result_file) {
    std::string full_base_name = file_name;
    auto pos = full_base_name.rfind("/");
    if (pos != std::string::npos) {
        full_base_name = full_base_name.substr(pos, full_base_name.size());
    }
    spdlog::filename_t file_base_name;
    spdlog::filename_t ext;
    std::tie(file_base_name, ext) = spdlog::details::file_helper::split_by_extension(full_base_name);

    const spdlog::filename_t &bak_file_name = file_base_name + ext + "_" ;
    RenameFile(file_name, file_path + bak_file_name);

    const std::string &tmp_zip_file = file_path + "/" + "tmp_" + appid + ".zip";
    if (spdlog::details::os::path_exists(tmp_zip_file)){
        (void)spdlog::details::os::remove(tmp_zip_file);
    }
    const spdlog::filename_t &cmpr_file = GetAbsolutePath(file_path) + "/" + bak_file_name;
    zipper::Zipper zipfile(tmp_zip_file);
    zipfile.add(cmpr_file);
    zipfile.close();

    const std::string &zip_file = file_path + "/" + file_base_name + ".zip";
    RenameFile(tmp_zip_file, zip_file);

    (void)spdlog::details::os::remove(cmpr_file);

    zip_result_file = std::move(zip_file);

    return true;
}

} // hozon
} // netaos
} // logcollector