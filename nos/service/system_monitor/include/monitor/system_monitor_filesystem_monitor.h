#ifndef SYSTEM_MONITOR_FILESYSTEM_MONITOR_H
#define SYSTEM_MONITOR_FILESYSTEM_MONITOR_H

#include <inttypes.h>
#include <fstream>
#include <vector>
#include <unordered_map>
#include "json/json.h"
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

struct MonitorInfo
{
    std::string md5;
    std::string rwpermission;
    std::string user;
    std::string group;
};


class SystemMonitorFileSystemMonitor : public SystemMonitorBase {
public:
    SystemMonitorFileSystemMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorFileSystemMonitor();

    virtual void Start();
    virtual void Stop();

private:

    void InitSocStatus();
    void DeInitSocStatus();
    // file protect
    void CreateProtectConfigFile();
    void CheckFileProtectChange();

    // file monitor
    void CreateMonitorConfigFile();
    void CheckMonitorConfig();
    void FileMonitor();

    void ReadConfPathInfo(const std::string id);
    void GetFileName(const char* path, const std::string id, const bool isRecursive=false);
    std::string GetFileMD5(const std::string& path);
    char* GetJsonAll(const char *fname);
    std::string GetPathRWPermission(const std::string& path);
    std::string GetPathUserId(const std::string& path);
    std::string GetPathGroupId(const std::string& path);

    void ReadFile(const std::string path, std::string& data);
    void WriteFile(const std::string path, const Json::Value data);
    void WriteFile(const std::string path, const std::string data);

    void OnSocStatusCallback(const std::string& domain, const std::string& key, const std::string& value);

    // delete json file and verify file
    void DeleteFile(const std::string fileName);

private:
    std::ofstream write_ofs_;
    bool stop_flag_;
    std::unordered_map<std::string, MonitorInfo> file_protect_path_map_;
    std::unordered_map<std::string, MonitorInfo> file_monitor_file_map_;
    std::unordered_map<std::string, MonitorInfo> file_monitor_path_map_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_FILESYSTEM_MONITOR_H
