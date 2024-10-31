//
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
//
/// @file log_file_remote_zipper_compression.cpp
/// @brief
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/log_file_remote_zipper_compression.h"

#include "spdlog/common.h"
#include "spdlog/details/file_helper.h"
#include "zipper.h"
#include "zmq_ipc/proto/log_server.pb.h"

namespace hozon {
namespace netaos {
namespace logcollector {

LogFileRemoteZipperCompression::LogFileRemoteZipperCompression(const std::string &cmpr_log_service_name) :
    LogFileCompressionBase(),
    compression_log_service_name_(cmpr_log_service_name) {
        client_ = std::make_unique<hozon::netaos::zmqipc::ZmqIpcClient>();
        client_->Init(compression_log_service_name_);
}

LogFileRemoteZipperCompression::~LogFileRemoteZipperCompression() {
    if (client_) {
        client_->Deinit();
    }
}

bool LogFileRemoteZipperCompression::DO(const std::string &appid,
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
    RenameFile(file_name, file_path + "/" + bak_file_name);

    const std::string &tmp_zip_file = file_path + "tmp_" + appid + ".zip";
    if (spdlog::details::os::path_exists(tmp_zip_file)){
        (void)spdlog::details::os::remove(tmp_zip_file);
    }

    const std::string &abs_file_path = GetAbsolutePath(file_path);
    const spdlog::filename_t &cmpr_file = abs_file_path + "/" + bak_file_name;

    try {
        ZipperInfo info{};
        info.set_file_path(abs_file_path);
        info.set_file_base_name(appid);
        info.set_filename(cmpr_file);
        info.set_basename(file_base_name);

        std::string serializedData = info.SerializeAsString();
        errno = 0;
        client_->RequestAndForget(serializedData);
        if (errno == 11) {
            errno = 0;
            zipper::Zipper zipfile(tmp_zip_file);
            zipfile.add(cmpr_file);
            zipfile.close();
        }
    }
    catch (const zmq::error_t& ex) {
        std::cout << " zmq Request error" << std::endl;
        zipper::Zipper zipfile(tmp_zip_file);
        zipfile.add(cmpr_file);
        zipfile.close();
    }

    zip_result_file = abs_file_path + file_base_name + ".zip";

    return true;
}

} // namespace logcollector
} // namespace netaos
} // namespace hozon
