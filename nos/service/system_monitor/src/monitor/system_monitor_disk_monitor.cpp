#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include <dirent.h>
#include <sys/vfs.h>
#include <regex>
#include <sys/stat.h>
#include "system_monitor/include/monitor/system_monitor_disk_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

// 调用linux系统命令获取disk信息
#define DISK_CMD "df -h | sed -n '2,$p'"
const std::string DISK_MONITOR_RECORD_FILE_NAME = "disk_monitor.log";
const uint32_t DISK_MORE_THAN_90_FAULT = 407001;
// log move
const std::string SOC_LOG_FILE_PATH = "/opt/usr/log/soc_log";
const std::string SOC_LOG_FILE_MOVE_PATH = "/opt/usr/log_bak/soc_log";
const std::string MCU_LOG_FILE_PATH = "/opt/usr/log/mcu_log";
const std::string MCU_LOG_FILE_MOVE_PATH = "/opt/usr/log_bak/mcu_log";

SystemMonitorDiskMonitor::SystemMonitorDiskMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, DISK_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
{
}

SystemMonitorDiskMonitor::~SystemMonitorDiskMonitor()
{
}

void
SystemMonitorDiskMonitor::Start()
{
    STMM_INFO << "SystemMonitorDiskMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_disk([this]() {
        while (!stop_flag_) {
            std::vector<DiskInfo> diskStatus;
            int32_t res = GetDiskStatus(diskStatus);
            if (!(res > 0)) {
                STMM_WARN << "SystemMonitorDiskMonitor get disk info failed.";
                std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                continue;
            }

            char buffer[5000];
            uint i = sprintf(buffer, "Filesystem          Size    Used    Avail    Use     MountedOn\n");
            for (auto& item : diskStatus) {
                i += sprintf(buffer + i, "%-17s %6s %7s %7s %7s     %s\n",
                             item.fileSystem, item.size, item.used, item.avail, item.use, item.mountedOn);
            }

            std::string notifyStr = CONTROL_OUTPUT_LINE + buffer + CONTROL_OUTPUT_LINE;
            Notify(notifyStr);
            std::string alarmStr = "";
            uint8_t faultStatus = 0;
            std::vector<std::string> partitionList;
            partitionList.clear();
            for (auto& item : diskStatus) {
                std::string useage = "";
                for (auto& ch : item.use) {
                    if ((ch >= '0' && ch <= '9')) {
                        useage += ch;
                    }
                }

                uint8_t usage = static_cast<uint8_t>(std::strtoul(useage.c_str(), 0, 10));
                if (SystemMonitorConfig::getInstance()->IsDiskMonitorPathList(static_cast<std::string>(item.mountedOn))) {
                    uint8_t maxAlarmValue = GetAlarmValue();
                    if (usage > maxAlarmValue) {
                        partitionList.push_back(static_cast<std::string>(item.mountedOn));
                        alarmStr += "The partition: " + static_cast<std::string>(item.mountedOn) + " usage: " + useage.c_str()
                                 + "% is too high and has exceeded the maxAlarmValue: " + std::to_string(maxAlarmValue) + "%.\n";
                    }
                    else {
                        uint8_t partitionAlarmValue = SystemMonitorConfig::getInstance()->GetPartitionAlarmValue(static_cast<std::string>(item.mountedOn));
                        if (usage > partitionAlarmValue) {
                            partitionList.push_back(static_cast<std::string>(item.mountedOn));
                            alarmStr += "The partition: " + static_cast<std::string>(item.mountedOn) + " usage: " + useage.c_str()
                                    + "% is too high and has exceeded the partitionAlarmValue: " + std::to_string(partitionAlarmValue) + "%.\n";
                        }
                    }
                }
            }

            if ("" != alarmStr) {
                Alarm(alarmStr);
                faultStatus = 1;
            }

            ReportFault(DISK_MORE_THAN_90_FAULT, faultStatus);
            SetRecordStr(notifyStr + alarmStr);
            if ("on" == GetPostProcessingSwitch()) {
                PostProcessing(partitionList);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_disk.native_handle(), "stmm_disk");
    stmm_disk.detach();
}

void
SystemMonitorDiskMonitor::Stop()
{
    STMM_INFO << "SystemMonitorDiskMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

int32_t
SystemMonitorDiskMonitor::GetDiskStatus(std::vector<DiskInfo>& diskStatus)
{
    FILE *fp = popen(DISK_CMD, "r");
    if (fp == NULL) {
        return -1;
    }

    DiskInfo item;
    diskStatus.clear();
    while (fscanf(fp, "%s %s %s %s %s %s",
        item.fileSystem, item.size, item.used, item.avail, item.use, item.mountedOn) == 6) {
        diskStatus.push_back(item);
    }

    pclose(fp);
    return diskStatus.size();
}

void
SystemMonitorDiskMonitor::PostProcessing(const std::vector<std::string>& partitionList)
{
    if (partitionList.size() <= 0) {
        return;
    }

    auto diskMonitorPathList = SystemMonitorConfig::getInstance()->GetDiskMonitorPathList();
    for (auto& partition : partitionList) {
        // 判断是否支持特殊后处理
        if (diskMonitorPathList[partition].isIncludeSpecialPostProcessing) {
            SpecialPostProcessing(partition);
        }

        for (auto& list : diskMonitorPathList[partition].deleteFilesByWildcardList) {
            // 判断分区剩余大小，若满足要求直接break
            if (GetPartitionAvailableSize(partition) >= diskMonitorPathList[partition].reservedSize) {
                break;
            }

            auto vec = Split(list.pathName);
            if (vec.size() <= 1) {
                continue;
            }

            std::string dirPath = "";
            for (uint i = 1; i < (vec.size() - 1); i++) {
                dirPath += ("/" + vec[i]);
            }

            std::vector<std::string> wildcards;
            std::string str = "";
            for (auto& ch : vec[vec.size() - 1]) {
                if ('*' == ch) {
                    if ("" != str) {
                        wildcards.push_back(str);
                        str = "";
                        continue;
                    }
                }
                else {
                    str += ch;
                }
            }

            if ("" != str) {
                wildcards.push_back(str);
            }

            DeleteFiles(partition, dirPath, diskMonitorPathList[partition].reservedSize, diskMonitorPathList[partition].isDeleteEmptyDir, list.isTraverseSubdir, wildcards);
        }

        for (auto& list : diskMonitorPathList[partition].deleteFilesByPathList) {
            // 判断分区剩余大小，若满足要求直接break
            if (GetPartitionAvailableSize(partition) >= diskMonitorPathList[partition].reservedSize) {
                break;
            }

            DeleteFiles(partition, list.pathName, diskMonitorPathList[partition].reservedSize, diskMonitorPathList[partition].isDeleteEmptyDir, list.isTraverseSubdir);
        }
    }
}

void
SystemMonitorDiskMonitor::DeleteFiles(const std::string& partition, const std::string& path, const uint32_t reservedSize,
                                      const bool isDeleteDir, const bool isTraverseSubdir, const std::vector<std::string>& wildcards)
{
    if (0 != access(path.c_str(), F_OK)) {
        return;
    }

    DIR* directory = opendir(path.c_str());
    if (!directory) {
        std::remove(path.c_str());
        return;
    }

    dirent* entry;
    while ((entry = readdir(directory)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        std::string sub_path = std::string(path) + "/" + entry->d_name;
        if ((DT_DIR == entry->d_type) && isTraverseSubdir) {
            // 子目录递归
            DeleteFiles(partition, sub_path, reservedSize, isDeleteDir, isTraverseSubdir, wildcards);
        }
        else {
            // 通配符检查
            if (wildcards.size()) {
                bool misMatch = false;
                for (auto& item : wildcards) {
                    if (std::string::npos == sub_path.find(item)) {
                        misMatch = true;
                        break;
                    }
                }

                if (misMatch) {
                    continue;
                }
            }

            // 判断分区剩余大小，若满足要求直接return
            if (GetPartitionAvailableSize(partition) >= reservedSize) {
                closedir(directory);
                return;
            }

            std::remove(sub_path.c_str());
        }
    }

    closedir(directory);
    if (isDeleteDir) {
        // 删除空目录
        std::remove(path.c_str());
    }
}

float
SystemMonitorDiskMonitor::GetPartitionAvailableSize(const std::string& partition)
{
    struct statfs diskInfo;
	statfs(partition.c_str(), &diskInfo);

	unsigned long long blocksize = diskInfo.f_bsize;                         // 每个block里面包含的字节数
	// unsigned long long totalsize = blocksize*diskInfo.f_blocks;           // 总的字节数
	unsigned long long availablesize = diskInfo.f_bavail * blocksize;        // 可用空间

    return static_cast<float>((availablesize >> 10) / 1024);   // printf("TOTAL_SIZE == %llu KB  %llu MB  %llu GB\n",totalsize>>10,totalsize>>20,totalsize>>30); // 分别换成KB,MB,GB为单位
}

void
SystemMonitorDiskMonitor::SpecialPostProcessing(const std::string& partition)
{
    if ("/opt/usr/log" == partition) {
        // 首先判断是否有异常文件（大于200m），如果有删除异常文件后直接return
        std::vector<std::string> files;
        SystemMonitorDiskMonitorGetFilesTypeInfo typeInfo;
        typeInfo.type = SystemMonitorDiskMonitorGetFilesType::kSizeExceedsLimit;
        typeInfo.value = "209715200";
        GetFiles(typeInfo, SOC_LOG_FILE_PATH, files);
        GetFiles(typeInfo, MCU_LOG_FILE_PATH, files);
        if (files.size()) {
            for (auto& item : files) {
                std::remove(item.c_str());
                STMM_ERROR << "SystemMonitorDiskMonitor::SpecialPostProcessing partition: " << partition << " delete abnormal file: " << item;
            }

            return;
        }

        // soc log move
        LogMove(SystemMonitorDiskMonitorLogMoveType::kSoc);
        // mcu log move
        LogMove(SystemMonitorDiskMonitorLogMoveType::kMcu);
    }
    else if ("/opt/usr/data" == partition) {
        // 删除coredump目录以外的所有文件
        DeleteFilesExceptCoredump(partition);
    }
}

void
SystemMonitorDiskMonitor::LogMove(const SystemMonitorDiskMonitorLogMoveType type)
{
    // 根据类型确定log文件路径和移动的目标路径
    std::string logFilePath = "";
    std::string logFileMovePath = "";
    switch(type)
    {
        case SystemMonitorDiskMonitorLogMoveType::kSoc:
            logFilePath = SOC_LOG_FILE_PATH;
            logFileMovePath = SOC_LOG_FILE_MOVE_PATH;
            break;
        case SystemMonitorDiskMonitorLogMoveType::kMcu:
            logFilePath = MCU_LOG_FILE_PATH;
            logFileMovePath = MCU_LOG_FILE_MOVE_PATH;
            break;
        default:
            break;
    }

    if ("" == logFilePath || "" == logFileMovePath) {
        return;
    }

    // 遍历文件名前缀
    auto logMoveList = SystemMonitorConfig::getInstance()->GetDiskMonitorLogMoveList(type);
    std::vector<std::string> files;
    SystemMonitorDiskMonitorGetFilesTypeInfo typeInfo;
    typeInfo.type = SystemMonitorDiskMonitorGetFilesType::kFixedPrefix;
    for (auto& item : logMoveList) {
        files.clear();
        typeInfo.value = item.filePrefix;
        GetFiles(typeInfo, logFilePath, files);
        if (files.size() <= item.reservedFileNum) {
            continue;
        }

        // 根据文件修改时间排序
        // std::sort(files.begin(), files.end(), [&](const std::string& a, const std::string& b) {
        //     struct stat fileStatA, fileStatB;
        //     stat(a.c_str(), &fileStatA);
        //     stat(b.c_str(), &fileStatB);
        //     return fileStatA.st_mtime < fileStatB.st_mtime;
        // });

        // 根据文件名中的序号排序
        std::sort(files.begin(), files.end(), [&](const std::string& a, const std::string& b) {
            // 定义正则表达式匹配序号部分
            std::regex pattern(R"(\d+)");
            // 提取文件名中的序号部分
            std::smatch match1, match2;
            std::regex_search(a, match1, pattern);
            std::regex_search(b, match2, pattern);
            // 将提取的序号部分转换为整数进行比较
            int num1 = std::stoi(match1.str());
            int num2 = std::stoi(match2.str());
            if (std::abs(num1 - num2) > 1000) {
                return num1 > num2;
            }

            return num1 < num2;
        });

        // 移动旧文件
        std::string srcFileName = "";
        std::string desFileName = "";
        for (uint i = 0; i < (files.size() - item.reservedFileNum); i++) {
            srcFileName = files[i];
            auto vec = Split(srcFileName, "/");
            desFileName = logFileMovePath + "/" + *(vec.end() - 1);
            // 由于在orin上使用rename进行文件移动一直失败，返回值-1，不知道是不是log文件权限导致，这里采用先复制再删除
            if (CopyFile(srcFileName, desFileName)) {
                std::remove(srcFileName.c_str());
            }
        }
    }
}

void
SystemMonitorDiskMonitor::GetFiles(const SystemMonitorDiskMonitorGetFilesTypeInfo& typeInfo, const std::string& dirPath, std::vector<std::string>& files)
{
    if (0 != access(dirPath.c_str(), F_OK)) {
        return;
    }

    DIR* directory = opendir(dirPath.c_str());
    if (nullptr == directory) {
        return;
    }

    dirent* entry;
    struct stat fileStat;
    std::string filename = "";
    std::string filepath = "";
    while ((entry = readdir(directory)) != nullptr) {
        filename = entry->d_name;
        filepath = dirPath + "/" + filename;
        if (stat(filepath.c_str(), &fileStat) == -1) {
            continue;
        }

        if (S_ISDIR(fileStat.st_mode)) {
            if (filename != "." && filename != "..") {
                GetFiles(typeInfo, filepath, files);
            }
        }
        else if (S_ISREG(fileStat.st_mode)) {
            if (SystemMonitorDiskMonitorGetFilesType::kFixedPrefix == typeInfo.type) {
                if (typeInfo.value == filename.substr(0, typeInfo.value.size())) {
                    files.push_back(filepath);
                }
            }
            else if (SystemMonitorDiskMonitorGetFilesType::kSizeExceedsLimit == typeInfo.type) {
                uint64_t value = static_cast<uint64_t>(std::strtoul(typeInfo.value.c_str(), 0, 10));
                if (static_cast<uint64_t>(fileStat.st_size) > value) {
                    files.push_back(filepath);
                }
            }
        }
    }

    closedir(directory);
}

void
SystemMonitorDiskMonitor::DeleteFilesExceptCoredump(const std::string& dirPath, const bool isDeleteDir)
{
    if (0 != access(dirPath.c_str(), F_OK)) {
        return;
    }

    if ("/opt/usr/data/coredump" == dirPath) {
        return;
    }

    DIR* directory = opendir(dirPath.c_str());
    if (nullptr == directory) {
        return;
    }

    dirent* entry;
    while ((entry = readdir(directory)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        std::string sub_path = std::string(dirPath) + "/" + entry->d_name;
        if (DT_DIR == entry->d_type) {
            // 子目录递归
            DeleteFilesExceptCoredump(sub_path, true);
        }
        else {
            std::remove(sub_path.c_str());
        }
    }

    closedir(directory);
    if (isDeleteDir) {
        // 删除空目录
        std::remove(dirPath.c_str());
    }
}

std::vector<std::string>
SystemMonitorDiskMonitor::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
