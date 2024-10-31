#ifndef SYSTEM_MONITOR_DISK_MONITOR_H
#define SYSTEM_MONITOR_DISK_MONITOR_H

#include <inttypes.h>
#include <string>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

struct DiskInfo {
    char fileSystem[100];
    char size[10];
    char used[10];
    char avail[10];
    char use[10];
    char mountedOn[100];
};

class SystemMonitorDiskMonitor : public SystemMonitorBase {
public:
    SystemMonitorDiskMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorDiskMonitor();

    virtual void Start();
    virtual void Stop();

private:
    int32_t GetDiskStatus(std::vector<DiskInfo>& diskStatus);

    // post processing
    void PostProcessing(const std::vector<std::string>& partitionList);
    void DeleteFiles(const std::string& partition, const std::string& path, const uint32_t reservedSize,
                     const bool isDeleteDir = false, const bool isTraverseSubdir = false,
                     const std::vector<std::string>& wildcards = {});
    float GetPartitionAvailableSize(const std::string& partition);

    // special post processing
    void SpecialPostProcessing(const std::string& partition);
    void LogMove(const SystemMonitorDiskMonitorLogMoveType type);
    void GetFiles(const SystemMonitorDiskMonitorGetFilesTypeInfo& typeInfo, const std::string& dirPath, std::vector<std::string>& files);
    void DeleteFilesExceptCoredump(const std::string& dirPath, const bool isDeleteDir = false);

    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = "/");

private:
    bool stop_flag_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_DISK_MONITOR_H
