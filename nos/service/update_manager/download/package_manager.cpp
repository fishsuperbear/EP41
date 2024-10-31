/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: ota package manager
 */
#include "update_manager/download/package_manager.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/file_to_bin/file_to_bin.h"
#include "update_manager/config/update_settings.h"

namespace hozon {
namespace netaos {
namespace update {


PackageManager::PackageManager()
{
}

PackageManager::~PackageManager()
{
}

int32_t PackageManager::Init()
{
    //TODO: Init secure comm lib
    UPDATE_LOG_D("Init");
    return 0;
}

int32_t PackageManager::Start()
{
    UPDATE_LOG_D("Start");
    return 0;
}

int32_t PackageManager::Stop()
{
    //TODO: Deinit secure comm lib
    UPDATE_LOG_D("Stop");
    return 0;
}

int32_t PackageManager::Deinit()
{
    //TODO: Deinit secure comm lib
    UPDATE_LOG_D("Deinit");
    return 0;
}

void
PackageManager::SetUrl(std::string& url)
{
    url_ = url;
}

void
PackageManager::RequestDownload(std::function<void(download_status_t*)> download_status_callback)
{
    //TODO: request download by secure comm lib
    download_status_callback_ = download_status_callback;
}

bool
PackageManager::IsDownloadInterrupt()
{
    //TODO: query download status by secure comm lib
    return true;
}

void
PackageManager::CancelDownload()
{
    //TODO: cancel download by secure comm lib
}

void
PackageManager::Verify(std::function<void(download_status_t*)> download_status_callback)
{
    //TODO: verify by secure comm lib
    download_status_callback_ = download_status_callback;
}

bool
PackageManager::HashCheck()
{
    //TODO: hash check by secure comm lib
    return true;
}

bool
PackageManager::PackageAnalyse()
{
    UPDATE_LOG_D("PackageAnalyse");
    return true;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon