#ifndef REMOTE_DIAG_DYNAMIC_PLUGIN_H
#define REMOTE_DIAG_DYNAMIC_PLUGIN_H

#include <mutex>
#include <iostream>
#include <unordered_map>

#include "remote_diag/include/common/remote_diag_def.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiagDynamicPlugin {

public:
    static RemoteDiagDynamicPlugin* getInstance();

    void Init();
    void DeInit();

    void RunPlugin(const RemoteDiagPluginRunInfo& pluginInfo);
    void GetPluginRunResult(const RemoteDiagPluginRunResultInfo& resultInfo);
    void CheckPluginResult(const std::string& filePath);

private:
    RemoteDiagDynamicPlugin();
    RemoteDiagDynamicPlugin(const RemoteDiagDynamicPlugin &);
    RemoteDiagDynamicPlugin & operator = (const RemoteDiagDynamicPlugin &);

private:
    char* GetJsonAll(const char *fname);
    void ParsePluginDescribeJson(const std::string& pluginPath);
    bool PluginPackageVerification(const std::string& pluginPackagePath);
    void RemovePluginDirectory(const std::string& pluginDirPath);

private:
    static RemoteDiagDynamicPlugin* instance_;
    static std::mutex mtx_;

    bool stop_flag_;

    std::string plugin_package_dir_path_;
    std::string current_plugin_name_;
    // unordered_map<pluginName, RemoteDiagPluginDescribeInfo>
    std::unordered_map<std::string, RemoteDiagPluginDescribeInfo> plugin_describe_info_map_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // #define REMOTE_DIAG_DYNAMIC_PLUGIN_H