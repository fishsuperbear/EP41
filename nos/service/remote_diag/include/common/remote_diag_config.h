/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: remote diag config
 */

#ifndef REMOTE_DIAG_CONFIG_H
#define REMOTE_DIAG_CONFIG_H

#include <mutex>
#include <algorithm>

#include "remote_diag/include/common/remote_diag_def.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiagConfig {
public:
    static RemoteDiagConfig* getInstance();

    void Init();
    void DeInit();

    void LoadRemoteDiagConfig();
    const RemoteDiagConfigInfo& GetRemoteDiagConfigInfo() {return remote_diag_config_info_;}

    // For Test
    void QueryPrintConfigData();

private:
    char* GetJsonAll(const char *fname);
    void ParseRemoteDiagConfigJson();
    std::string GetVinNumber();

    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr);

private:
    RemoteDiagConfig();
    RemoteDiagConfig(const RemoteDiagConfig &);
    RemoteDiagConfig & operator = (const RemoteDiagConfig &);

private:
    static std::mutex mtx_;
    static RemoteDiagConfig* instance_;

    // remote diag config info
    RemoteDiagConfigInfo remote_diag_config_info_;

    std::string vin_number_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // REMOTE_DIAG_CONFIG_H