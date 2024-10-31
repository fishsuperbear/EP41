/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: ota package manager
 */
#ifndef PACKAGE_MANAGER_H
#define PACKAGE_MANAGER_H

#include <stdint.h>
#include <string>
#include <functional>

#include "update_manager/common/data_def.h"
#include "update_manager/common/common_operation.h"

namespace hozon {
namespace netaos {
namespace update {


class PackageManager {
public:
    PackageManager();
    ~PackageManager();

    int32_t Init();
    int32_t Start();
    int32_t Stop();
    int32_t Deinit();

    void SetUrl(std::string& url);
    void RequestDownload(std::function<void(download_status_t*)> download_status_callback);
    void CancelDownload();
    bool IsDownloadInterrupt();
    void Verify(std::function<void(download_status_t*)> download_status_callback);
    bool HashCheck();
    bool PackageAnalyse();

private:
    PackageManager(const PackageManager &);
    PackageManager & operator = (const PackageManager &);

    std::function<void(download_status_t*)> download_status_callback_;
    std::string url_;

};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // PACKAGE_MANAGER_H