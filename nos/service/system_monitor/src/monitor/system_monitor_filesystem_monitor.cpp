#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <dirent.h>
#include <sys/stat.h>
#include <pwd.h>
#include <grp.h>
#include <cstring>
#include "cfg/include/config_param.h"
#include "system_monitor/include/common/md5.h"
#include "system_monitor/include/common/system_monitor_config.h"
#include "system_monitor/include/monitor/system_monitor_filesystem_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

using namespace hozon::netaos::cfg;

const int MAX_LOAD_SIZE = 1024;

const std::string FILE_PROTECT_TYPE = "file_protect";
const std::string FILE_MONITOR_TYPE = "file_monitor";
const std::string FILE_MONITOR_CONF_TYPE = "file_monitor_conf";

const std::string FILE_MONITOR_RECORD_FILE_NAME = "file_monitor.log";
const std::string FILE_PROTECT_LIST_PATH = "/cfg/conf_app/file_monitor/.file_protect_list.json";
const std::string FILE_PROTECT_VERIFY_PATH = "/cfg/conf_app/file_monitor/.file_protect_list_verify";
const std::string FILE_MONITOR_LIST_PATH = "/cfg/conf_app/file_monitor/.file_monitor_list.json";
const uint32_t FILE_CHANGED_FAULT = 408001;

SystemMonitorFileSystemMonitor::SystemMonitorFileSystemMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, FILE_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
{
}

SystemMonitorFileSystemMonitor::~SystemMonitorFileSystemMonitor()
{
}

void
SystemMonitorFileSystemMonitor::Start()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::Start";
    InitSocStatus();
    file_protect_path_map_.clear();
    file_monitor_file_map_.clear();
    file_monitor_path_map_.clear();
    // file protect(every startup check)
    if (0 != access(FILE_PROTECT_LIST_PATH.c_str(), F_OK)) {
        STMM_INFO << "SystemMonitorFileSystemMonitor::Start | file protect " << FILE_PROTECT_LIST_PATH << " not exist";
        // First startup after upgrading
        CreateProtectConfigFile();
    }
    else {
        // Every startup
        CheckFileProtectChange();
    }

    // file monitor
    if (0 != access(FILE_MONITOR_LIST_PATH.c_str(), F_OK)) {
        STMM_INFO << "SystemMonitorFileSystemMonitor::Start | file monitor " << FILE_MONITOR_LIST_PATH << " not exist";
        CreateMonitorConfigFile();
    }
    else {
        CheckMonitorConfig();
    }

}

void
SystemMonitorFileSystemMonitor::Stop()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::Stop";
    stop_flag_ = true;
    file_protect_path_map_.clear();
    file_monitor_file_map_.clear();
    file_monitor_path_map_.clear();
    DeInitSocStatus();
}

void
SystemMonitorFileSystemMonitor::CreateProtectConfigFile()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::CreateProtectConfigFile";
    ReadConfPathInfo(FILE_PROTECT_TYPE);

    Json::Value data;
    data["count"] = file_protect_path_map_.size();
    uint i = 0;
    for (auto& item : file_protect_path_map_) {
        data["value"][i]["path"] = item.first;
        data["value"][i]["md5"] = item.second.md5;
        data["value"][i]["rwpermission"] = item.second.rwpermission;
        data["value"][i]["user"] = item.second.user;
        data["value"][i]["group"] = item.second.group;
        i++;
    }

    WriteFile(FILE_PROTECT_LIST_PATH, data);
    WriteFile(FILE_PROTECT_VERIFY_PATH, GetFileMD5(FILE_PROTECT_LIST_PATH));
}

void
SystemMonitorFileSystemMonitor::InitSocStatus()
{
    CfgResultCode res = ConfigParam::Instance()->Init(1000);
    if (CONFIG_OK != res) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::InitSocStatus init error";
        return;
    }

    ConfigParam::Instance()->MonitorParam<std::string>("system/soc_status", std::bind(&SystemMonitorFileSystemMonitor::OnSocStatusCallback,
        this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

void
SystemMonitorFileSystemMonitor::DeInitSocStatus()
{
    ConfigParam::Instance()->UnMonitorParam("system/soc_status");
}

void
SystemMonitorFileSystemMonitor::CheckFileProtectChange()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::CheckFileProtectChange";
    if ((0 != access(FILE_PROTECT_LIST_PATH.c_str(), F_OK)) ||
        (0 != access(FILE_PROTECT_VERIFY_PATH.c_str(), F_OK))) {
        STMM_INFO << "SystemMonitorFileSystemMonitor::CheckFileProtectChange Verification file does not exist!";
        return;
    }

    std::string verify_value = "";
    ReadFile(FILE_PROTECT_VERIFY_PATH, verify_value);

    if (GetFileMD5(FILE_PROTECT_LIST_PATH) != verify_value) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::CheckFileProtectChange FileProtect, Changes in file protection: list json has been modified!";
        SetRecordStr("Changes in file protection: list json has been modified!");
        WriteDataToFile();
        Alarm("Changes in file protection: list json has been modified!");
        ReportFault(FILE_CHANGED_FAULT, 1);
        return;
    }

    ReadConfPathInfo(FILE_PROTECT_TYPE);

    char* jsonstr = GetJsonAll(FILE_PROTECT_LIST_PATH.c_str());
    if (nullptr == jsonstr) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::CheckFileProtectChange error jsonstr is nullptr.";
        return;
    }

    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value  data;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &data, &errs);
    if (!res || !errs.empty()) {
        if (jsonstr != NULL) {
            free(jsonstr);
        }
        STMM_ERROR << "SystemMonitorFileSystemMonitor::CheckFileProtectChange jsonstr is nullptr.";
        return;
    }

    std::string file_add_name = "";
    std::string file_del_name = "";
    std::string file_modify_name = "";
    std::string file_rw_modify_name = "";
    std::string file_user_modify_name = "";
    std::string file_group_modify_name = "";
    Json::Value value = data["value"];
    for (Json::ArrayIndex i = 0; i < value.size(); ++i) {
        auto path_find = file_protect_path_map_.find(static_cast<std::string>(value[i]["path"].asString()));
        if (path_find != file_protect_path_map_.end()) {
            if (path_find->second.md5 != static_cast<std::string>(value[i]["md5"].asString())) {
                // file modify
                file_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                    + ", current md5: [" + path_find->second.md5
                                    + "], source md5: [" + static_cast<std::string>(value[i]["md5"].asString()) + "]\n";
            }

            if (path_find->second.rwpermission != static_cast<std::string>(value[i]["rwpermission"].asString())) {
                // file rw permission modify
                file_rw_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                       + ", current permission: [" + path_find->second.rwpermission
                                       + "], source permission: [" + static_cast<std::string>(value[i]["rwpermission"].asString()) + "]\n";
            }

            if (path_find->second.user != static_cast<std::string>(value[i]["user"].asString())) {
                // file user modify
                file_user_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                         + ", current user: [" + path_find->second.user
                                         + "], source user: [" + static_cast<std::string>(value[i]["user"].asString()) + "]\n";
            }

            if (path_find->second.group != static_cast<std::string>(value[i]["group"].asString())) {
                // file group modify
                file_group_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                          + ", current group: [" + path_find->second.group
                                          + "], source group: [" + static_cast<std::string>(value[i]["group"].asString()) + "]\n";
            }

            file_protect_path_map_.erase(path_find);
        }
        else {
            // file not exist
            file_del_name += "\t" + static_cast<std::string>(value[i]["path"].asString()) + "\n";
        }
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }

    if (!file_protect_path_map_.empty()) {
        // file add
        for (auto& item : file_protect_path_map_) {
            file_add_name += "\t" + item.first + "\n";
        }
    }

    std::string alarm_info = "";
    if (file_add_name != "") {
        alarm_info += "add file:\n" + file_add_name;
    }

    if (file_del_name != "") {
        alarm_info += "del file:\n" + file_del_name;
    }

    if (file_modify_name != "") {
        alarm_info += "modify file:\n" + file_modify_name;
    }

    if (file_rw_modify_name != "") {
        alarm_info += "rw permission modify file:\n" + file_rw_modify_name;
    }

    if (file_user_modify_name != "") {
        alarm_info += "user modify file:\n" + file_user_modify_name;
    }

    if (file_group_modify_name != "") {
        alarm_info += "group modify file:\n" + file_group_modify_name;
    }

    if (alarm_info != "") {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::CheckFileProtectChange FileProtect, Changes in file protection";
        SetRecordStr("Changes in file protection:\n" + alarm_info);
        WriteDataToFile();
        Alarm("Changes in file protection:\n" + alarm_info);
        ReportFault(FILE_CHANGED_FAULT, 1);
    }
}

void
SystemMonitorFileSystemMonitor::CreateMonitorConfigFile()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::CreateMonitorConfigFile";
    ReadConfPathInfo(FILE_MONITOR_TYPE);

    Json::Value data;
    data["count"] = file_monitor_path_map_.size();
    uint i = 0;
    for (auto& item : file_monitor_path_map_) {
        data["value"][i]["path"] = item.first;
        data["value"][i]["md5"] = item.second.md5;
        data["value"][i]["rwpermission"] = item.second.rwpermission;
        data["value"][i]["user"] = item.second.user;
        data["value"][i]["group"] = item.second.group;
        i++;
    }

    WriteFile(FILE_MONITOR_LIST_PATH, data);
    // WriteFile(FILE_PROTECT_VERIFY_PATH, GetFileMD5(FILE_MONITOR_LIST_PATH));

    FileMonitor();
}

void
SystemMonitorFileSystemMonitor::CheckMonitorConfig()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::CheckMonitorConfig";
    // read conf path
    std::vector<SystemMonitorFileMonitorInfo> confList;
    SystemMonitorConfig::getInstance()->GetFileMonitorPath(confList);

    char* jsonstr = GetJsonAll(FILE_MONITOR_LIST_PATH.c_str());
    if (nullptr == jsonstr) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::FileMonitor error jsonstr is nullptr.";
        return;
    }

    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value  data;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &data, &errs);
    if (!res || !errs.empty()) {
        if (jsonstr != NULL) {
            free(jsonstr);
        }
        STMM_ERROR << "SystemMonitorFileSystemMonitor::FileMonitor jsonstr is nullptr.";
        return;
    }

    std::unordered_map<std::string, MonitorInfo> file_map;
    MonitorInfo monitorinfo;
    Json::Value value = data["value"];
    for (Json::ArrayIndex i = 0; i < value.size(); ++i) {
        monitorinfo.md5 = static_cast<std::string>(value[i]["md5"].asString());
        monitorinfo.rwpermission = static_cast<std::string>(value[i]["rwpermission"].asString());
        monitorinfo.user = static_cast<std::string>(value[i]["user"].asString());
        monitorinfo.group = static_cast<std::string>(value[i]["group"].asString());
        file_map.insert(std::make_pair(static_cast<std::string>(value[i]["path"].asString()), monitorinfo));
    }

    file_monitor_file_map_.clear();
    for (auto item : confList) {
        auto path_find = file_map.find(item.pathName.c_str());
        if (path_find != file_map.end()) {
            continue;
        }
        else {
            GetFileName(item.pathName.c_str(), FILE_MONITOR_CONF_TYPE, item.isRecursive);
        }
    }

    for (auto& item : file_monitor_file_map_) {
        monitorinfo.md5 = item.second.md5;
        monitorinfo.rwpermission = item.second.rwpermission;
        monitorinfo.user = item.second.user;
        monitorinfo.group = item.second.group;
        file_map.insert(std::make_pair(item.first, monitorinfo));
    }

    Json::Value info;
    info["count"] = file_map.size();
    uint i = 0;
    for (auto& item : file_map) {
        info["value"][i]["path"] = item.first;
        info["value"][i]["md5"] = item.second.md5;
        info["value"][i]["rwpermission"] = item.second.rwpermission;
        info["value"][i]["user"] = item.second.user;
        info["value"][i]["group"] = item.second.group;
        i++;
    }

    WriteFile(FILE_MONITOR_LIST_PATH, info);

    FileMonitor();
}

void
SystemMonitorFileSystemMonitor::FileMonitor()
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::FileMonitor";
    stop_flag_ = false;
    std::thread stmm_file([this]() {
        std::string log_info = "";
        while (!stop_flag_) {
            ReadConfPathInfo(FILE_MONITOR_TYPE);
            char* jsonstr = GetJsonAll(FILE_MONITOR_LIST_PATH.c_str());
            if (nullptr == jsonstr) {
                STMM_ERROR << "SystemMonitorFileSystemMonitor::FileMonitor error jsonstr is nullptr.";
                return;
            }

            Json::CharReaderBuilder readerBuilder;
            std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
            Json::Value  data;
            JSONCPP_STRING errs;

            bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &data, &errs);
            if (!res || !errs.empty()) {
                if (jsonstr != NULL) {
                    free(jsonstr);
                }
                STMM_ERROR << "SystemMonitorFileSystemMonitor::FileMonitor jsonstr is nullptr.";
                return;
            }

            std::string file_add_name = "";
            std::string file_del_name = "";
            std::string file_modify_name = "";
            std::string file_rw_modify_name = "";
            std::string file_user_modify_name = "";
            std::string file_group_modify_name = "";
            Json::Value value = data["value"];
            for (Json::ArrayIndex i = 0; i < value.size(); ++i) {
                auto path_find = file_monitor_path_map_.find(static_cast<std::string>(value[i]["path"].asString()));
                if (path_find != file_monitor_path_map_.end()) {
                    if (path_find->second.md5 != static_cast<std::string>(value[i]["md5"].asString())) {
                        // file modify
                        file_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                            + ", current md5: [" + path_find->second.md5
                                            + "], source md5: [" + static_cast<std::string>(value[i]["md5"].asString()) + "]\n";
                    }

                    if (path_find->second.rwpermission != static_cast<std::string>(value[i]["rwpermission"].asString())) {
                        // file rw permission modify
                        file_rw_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                            + ", current permission: [" + path_find->second.rwpermission
                                            + "], source permission: [" + static_cast<std::string>(value[i]["rwpermission"].asString()) + "]\n";
                    }

                    if (path_find->second.user != static_cast<std::string>(value[i]["user"].asString())) {
                        // file user modify
                        file_user_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                                + ", current user: [" + path_find->second.user
                                                + "], source user: [" + static_cast<std::string>(value[i]["user"].asString()) + "]\n";
                    }

                    if (path_find->second.group != static_cast<std::string>(value[i]["group"].asString())) {
                        // file group modify
                        file_group_modify_name += "\t" + static_cast<std::string>(value[i]["path"].asString())
                                                + ", current group: [" + path_find->second.group
                                                + "], source group: [" + static_cast<std::string>(value[i]["group"].asString()) + "]\n";
                    }

                    file_monitor_path_map_.erase(path_find);
                }
                else {
                    // file not exist
                    file_del_name += "\t" + static_cast<std::string>(value[i]["path"].asString()) + "\n";
                }
            }

            if (jsonstr != NULL) {
                free(jsonstr);
            }

            if (!file_monitor_path_map_.empty()) {
                // file add
                for (auto& item : file_monitor_path_map_) {
                    file_add_name += "\t" + item.first + "\n";
                }
            }

            std::string alarm_info = "";
            if (file_add_name != "") {
                alarm_info += "add file:\n" + file_add_name;
            }

            if (file_del_name != "") {
                alarm_info += "del file:\n" + file_del_name;
            }

            if (file_modify_name != "") {
                alarm_info += "modify file:\n" + file_modify_name;
            }

            if (file_rw_modify_name != "") {
                alarm_info += "rw permission modify file:\n" + file_rw_modify_name;
            }

            if (file_user_modify_name != "") {
                alarm_info += "user modify file:\n" + file_user_modify_name;
            }

            if (file_group_modify_name != "") {
                alarm_info += "group modify file:\n" + file_group_modify_name;
            }

            if (alarm_info != "") {
                // STMM_ERROR << "SystemMonitorFileSystemMonitor::FileMonitor, Changes in file monitoring";
                if (log_info != alarm_info) {
                    SetRecordStr("Changes in file monitoring:\n" + alarm_info);
                    WriteDataToFile();
                    log_info = alarm_info;
                }
                // Alarm("Changes in file monitoring:\n" + alarm_info);
                // ReportFault(FILE_CHANGED_FAULT, 1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_file.native_handle(), "stmm_file");
    stmm_file.detach();
}

void
SystemMonitorFileSystemMonitor::ReadConfPathInfo(const std::string id)
{
    // read conf path
    std::vector<SystemMonitorFileMonitorInfo> confList;
    confList.clear();
    if (FILE_PROTECT_TYPE == id) {
        SystemMonitorConfig::getInstance()->GetFileProtectPath(confList);

        for (auto item : confList) {
            GetFileName(item.pathName.c_str(), FILE_PROTECT_TYPE, item.isRecursive);
        }
    }
    else if (FILE_MONITOR_TYPE == id) {
        SystemMonitorConfig::getInstance()->GetFileMonitorPath(confList);

        file_monitor_path_map_.clear();
        for (auto item : confList) {
            GetFileName(item.pathName.c_str(), FILE_MONITOR_TYPE, item.isRecursive);
        }

    }
}

void
SystemMonitorFileSystemMonitor::GetFileName(const char* path, const std::string id, const bool isRecursive)
{
    struct stat pathInfo;
    if (stat(path, &pathInfo) != 0) {
        STMM_INFO << "SystemMonitorFileSystemMonitor::GetFileName Error getting file information: " << path << ", id: " << id << ", isRecursive:" << isRecursive;
        return;
    }

    MonitorInfo monitorInfo;
    monitorInfo.rwpermission = GetPathRWPermission(path);
    monitorInfo.user = GetPathUserId(path);
    monitorInfo.group = GetPathGroupId(path);
    if (S_ISDIR(pathInfo.st_mode)) {
        monitorInfo.md5 = "";
        if (FILE_PROTECT_TYPE == id) {
            file_protect_path_map_.insert(std::make_pair(path, monitorInfo));
        }
        else if (FILE_MONITOR_TYPE == id) {
            file_monitor_path_map_.insert(std::make_pair(path, monitorInfo));
        }
        else if (FILE_MONITOR_CONF_TYPE == id) {
            file_monitor_file_map_.insert(std::make_pair(path, monitorInfo));
        }
        DIR* directory = opendir(path);
        if (directory == nullptr) {
            STMM_INFO << "SystemMonitorFileSystemMonitor::GetFileName Error opening directory: " << path << ", id: " << id;
            return;
        }

        struct dirent* entry;
        MonitorInfo pathMonitorInfo;
        while ((entry = readdir(directory)) != nullptr) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;

            // 构建完整路径
            std::string fullPath = std::string(path) + "/" + std::string(entry->d_name);

            struct stat fileInfo;
            if (stat(fullPath.c_str(), &fileInfo) != 0) {
                STMM_INFO << "SystemMonitorFileSystemMonitor::GetFileName Error getting file information: " << fullPath << ", id: " << id;
                continue;
            }

            pathMonitorInfo.rwpermission = GetPathRWPermission(fullPath);
            pathMonitorInfo.user = GetPathUserId(fullPath);
            pathMonitorInfo.group = GetPathGroupId(fullPath);
            if (S_ISDIR(fileInfo.st_mode)) {
                pathMonitorInfo.md5 = "";
                if (FILE_PROTECT_TYPE == id) {
                    file_protect_path_map_.insert(std::make_pair(fullPath, pathMonitorInfo));
                    if (isRecursive) {
                        GetFileName(fullPath.c_str(), FILE_PROTECT_TYPE, isRecursive);
                    }
                }
                else if (FILE_MONITOR_TYPE == id) {
                    file_monitor_path_map_.insert(std::make_pair(fullPath, pathMonitorInfo));
                    if (isRecursive) {
                        GetFileName(fullPath.c_str(), FILE_MONITOR_TYPE, isRecursive);
                    }
                }
                else if (FILE_MONITOR_CONF_TYPE == id) {
                    file_monitor_file_map_.insert(std::make_pair(fullPath, pathMonitorInfo));
                    if (isRecursive) {
                        GetFileName(fullPath.c_str(), FILE_MONITOR_CONF_TYPE, isRecursive);
                    }
                }
            }
            else {
                pathMonitorInfo.md5 = GetFileMD5(fullPath);
                if (FILE_PROTECT_TYPE == id) {
                    file_protect_path_map_.insert(std::make_pair(fullPath, pathMonitorInfo));
                }
                else if (FILE_MONITOR_TYPE == id) {
                    file_monitor_path_map_.insert(std::make_pair(fullPath, pathMonitorInfo));
                }
                else if (FILE_MONITOR_CONF_TYPE == id) {
                    file_monitor_file_map_.insert(std::make_pair(fullPath, pathMonitorInfo));
                }
            }
        }
    }
    else {
        monitorInfo.md5 = GetFileMD5(path);
        if (FILE_PROTECT_TYPE == id) {
            file_protect_path_map_.insert(std::make_pair(path, monitorInfo));
        }
        else if (FILE_MONITOR_TYPE == id) {
            file_monitor_path_map_.insert(std::make_pair(path, monitorInfo));
        }
        else if (FILE_MONITOR_CONF_TYPE == id) {
            file_monitor_file_map_.insert(std::make_pair(path, monitorInfo));
        }
    }
}

std::string
SystemMonitorFileSystemMonitor::GetFileMD5(const std::string& path)
{
    // STMM_INFO << "SystemMonitorFileSystemMonitor::GetFileMD5 path: " << path;
    if (0 != access(path.c_str(), F_OK)) {
        // STMM_WARN << "SystemMonitorFileSystemMonitor::GetFileMD5 path: " << path << " not exist.";
        return "";
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        STMM_WARN << "SystemMonitorFileSystemMonitor::GetFileMD5 path: " << path << " open failed.";
        return "";
    }

    MD5 md5;
    md5.reset();
    md5.update(in);
    std::string md5Value = md5.toString();
    // STMM_INFO << "SystemMonitorFileSystemMonitor::GetFileMD5 path: " << path << " MD5: " << md5Value;
    return md5Value;
}

char*
SystemMonitorFileSystemMonitor::GetJsonAll(const char *fname)
{
    FILE *fp;
    char *str;
    char txt[MAX_LOAD_SIZE];
    int filesize;
    if ((fp = fopen(fname, "r")) == NULL) {
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);

    str = (char*)malloc(filesize + 1);
    memset(str, 0, filesize);

    rewind(fp);
    while ((fgets(txt, MAX_LOAD_SIZE, fp)) != NULL) {
        strcat(str, txt);
    }
    fclose(fp);

    return str;
}

std::string
SystemMonitorFileSystemMonitor::GetPathRWPermission(const std::string& path)
{
    // STMM_INFO << "SystemMonitorFileSystemMonitor::GetPathRWPermission path: " << path;
    struct stat fileStat;

    if (stat(path.c_str(), &fileStat) != 0) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::GetPathRWPermission, Failed to get file/directory metadata: " << path;
        return "";
    }

    // 获取权限信息
    mode_t permissions = fileStat.st_mode;
    std::string rw_str = "";
    rw_str = ((permissions & S_IRUSR) ? "r" : "-");
    rw_str += ((permissions & S_IWUSR) ? "w" : "-");
    rw_str += ((permissions & S_IXUSR) ? "x" : "-");
    rw_str += ((permissions & S_IRGRP) ? "r" : "-");
    rw_str += ((permissions & S_IWGRP) ? "w" : "-");
    rw_str += ((permissions & S_IXGRP) ? "x" : "-");
    rw_str += ((permissions & S_IROTH) ? "r" : "-");
    rw_str += ((permissions & S_IWOTH) ? "w" : "-");
    rw_str += ((permissions & S_IXOTH) ? "x" : "-");

    return rw_str;
}

std::string
SystemMonitorFileSystemMonitor::GetPathUserId(const std::string& path)
{
    // STMM_INFO << "SystemMonitorFileSystemMonitor::GetPathUserId path: " << path;
    struct stat fileStat;

    if (stat(path.c_str(), &fileStat) != 0) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::GetPathUserId, Failed to get file/directory metadata: " << path;
        return "";
    }

    struct passwd *userInfo = getpwuid(fileStat.st_uid);
    if (userInfo == nullptr) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::GetPathUserId, Failed to get user information: " << path;
        return "";
    }

    return userInfo->pw_name;
}

std::string
SystemMonitorFileSystemMonitor::GetPathGroupId(const std::string& path)
{
    // STMM_INFO << "SystemMonitorFileSystemMonitor::GetPathGroupId path: " << path;
    struct stat fileStat;

    if (stat(path.c_str(), &fileStat) != 0) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::GetPathGroupId, Failed to get file/directory metadata: " << path;
        return "";
    }

    struct group *groupInfo = getgrgid(fileStat.st_gid);
    if (groupInfo == nullptr) {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::GetPathGroupId, Failed to get group information: " << path;
        return "";
    }

    return groupInfo->gr_name;
}

void
SystemMonitorFileSystemMonitor::ReadFile(const std::string path, std::string& data)
{
    std::ifstream ifs;
    data = "";
    ifs.open(path, std::ios::in);
    if (ifs.is_open()) {
        std::stringstream stream;
        stream << ifs.rdbuf();
        data = stream.str();
        ifs.close();
    }
    else {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::ReadFile, " << path << "open failed!";
    }
}

void
SystemMonitorFileSystemMonitor::WriteFile(const std::string path, const Json::Value data)
{
    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary);
    if (ofs.is_open()) {
        ofs << data;
        ofs.close();
    }
    else {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::WriteFile json, " << path << "open failed!";
    }
}

void
SystemMonitorFileSystemMonitor::WriteFile(const std::string path, const std::string data)
{
    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary);
    if (ofs.is_open()) {
        ofs << data;
        ofs.close();
    }
    else {
        STMM_ERROR << "SystemMonitorFileSystemMonitor::WriteFile string, " << path << "open failed!";
    }
}

void
SystemMonitorFileSystemMonitor::OnSocStatusCallback(const std::string& domain, const std::string& key, const std::string& value)
{
    STMM_INFO << "SystemMonitorFileSystemMonitor::OnSocStatusCallback domain: " << domain << ", key: " << key << ", value: " << value;
    if ("Update" == value) {
        DeleteFile(FILE_PROTECT_LIST_PATH);
        DeleteFile(FILE_PROTECT_VERIFY_PATH);
        DeleteFile(FILE_MONITOR_LIST_PATH);
    }
}

void
SystemMonitorFileSystemMonitor::DeleteFile(const std::string fileName)
{
    std::ifstream file(fileName);

    if (!file.is_open()) {
        return;
    }

    file.close();
    remove(fileName.c_str());
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
