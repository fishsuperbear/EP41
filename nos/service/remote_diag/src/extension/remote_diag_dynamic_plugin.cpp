/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: remote diag extension dynamic plugin
*/

#include <thread>
#include <fstream>
#include <sstream>
#include <dirent.h>

#include "json/json.h"
#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/handler/remote_diag_handler.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/extension/remote_diag_dynamic_plugin.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

RemoteDiagDynamicPlugin* RemoteDiagDynamicPlugin::instance_ = nullptr;
std::mutex RemoteDiagDynamicPlugin::mtx_;

const std::string RUN_FINISH_FILE_NAME = "done";
const int MAX_LOAD_SIZE = 5000000;

RemoteDiagDynamicPlugin::RemoteDiagDynamicPlugin()
: stop_flag_(false)
, plugin_package_dir_path_("")
, current_plugin_name_("")
{
}

RemoteDiagDynamicPlugin*
RemoteDiagDynamicPlugin::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new RemoteDiagDynamicPlugin();
        }
    }

    return instance_;
}

void
RemoteDiagDynamicPlugin::Init()
{
    DGR_INFO << "RemoteDiagDynamicPlugin::Init";
    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    plugin_package_dir_path_ = configInfo.FileDownloadDirPath;
}

void
RemoteDiagDynamicPlugin::DeInit()
{
    DGR_INFO << "RemoteDiagDynamicPlugin::DeInit";
    stop_flag_ = true;
    // stop test script
    for (auto& item : plugin_describe_info_map_) {
        if (RemoteDiagPluginRunStatusType::kDefault != item.second.runStatus) {
            std::string command = "cd " + item.second.scriptPath + "\n" + "bash stop.sh &";
            int ret = system(command.c_str());
            if (0 != ret) {
                DGR_WARN << "RemoteDiagDynamicPlugin::DeInit pluginName: " << item.second.pluginName << " stop failed.";
                continue;
            }

            DGR_INFO << "RemoteDiagDynamicPlugin::DeInit pluginName: " << item.second.pluginName << " stop succeed.";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    plugin_describe_info_map_.clear();
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }

    DGR_INFO << "RemoteDiagDynamicPlugin::DeInit finish!";
}

void
RemoteDiagDynamicPlugin::RunPlugin(const RemoteDiagPluginRunInfo& pluginInfo)
{
    DGR_INFO << "RemoteDiagDynamicPlugin::RunPlugin pluginPackageName: " << pluginInfo.pluginPackageName;
    std::thread([this](const RemoteDiagPluginRunInfo& pluginInfo) {
        Json::Value respMessage;
        respMessage["SA"] = pluginInfo.ta;
        respMessage["TA"] = pluginInfo.sa;
        respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kPluginRun];
        std::string plugin_packge_path = plugin_package_dir_path_ + pluginInfo.pluginPackageName;
        if (access(plugin_packge_path.c_str(), F_OK) != 0) {
            respMessage["DATA"] = "Plugin package does not exist!";
            RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
            return;
        }

        current_plugin_name_ = "";
        for (auto& item : pluginInfo.pluginPackageName) {
            if ('.' == item) {
                break;
            }

            current_plugin_name_ += item;
        }

        if (!PluginPackageVerification(plugin_packge_path)) {
            respMessage["DATA"] = "Plugin package verification failed!";
            RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
            return;
        }

        auto itr_find = plugin_describe_info_map_.find(current_plugin_name_);
        if (itr_find == plugin_describe_info_map_.end()) {
            respMessage["DATA"] = "Plugin package load describe info failed!.";
            RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
            return;
        }

        std::string plugin_path = plugin_package_dir_path_ + current_plugin_name_;
        DGR_INFO << "RemoteDiagDynamicPlugin::RunPlugin plugin_path: " << plugin_path;
        std::string plugin_script_path = plugin_path + "/script";
        std::string plugin_result_path = plugin_path + "/result";
        // Determine if the completed file exists, delete if it already exists
        if (access((plugin_result_path + "/" + RUN_FINISH_FILE_NAME).c_str(), F_OK) == 0) {
            remove((plugin_result_path + "/" + RUN_FINISH_FILE_NAME).c_str());
        }

        DGR_INFO << "RemoteDiagDynamicPlugin::RunPlugin pluginName: " << plugin_describe_info_map_[current_plugin_name_].pluginName << " pluginDescribe: " << plugin_describe_info_map_[current_plugin_name_].pluginDescribe
                                          << " pluginRunParameters: " << plugin_describe_info_map_[current_plugin_name_].pluginRunParameters << " runTimeout: " << plugin_describe_info_map_[current_plugin_name_].runTimeout
                                          << " isNeedCompressResult: " << plugin_describe_info_map_[current_plugin_name_].isNeedCompressResult << " unCompressFile: " << plugin_describe_info_map_[current_plugin_name_].unCompressFile;
        std::string command = "cd " + plugin_script_path + "\n" + "bash start.sh " + plugin_describe_info_map_[current_plugin_name_].pluginRunParameters + " &";
        int ret = system(command.c_str());
        if (0 != ret) {
            respMessage["DATA"] = "Plugin executed failed!";
            RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
            return;
        }

        RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_PLUGIN_RUN);
        RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(pluginInfo.sa.c_str(), 0, 16)), true);
        plugin_describe_info_map_[current_plugin_name_].scriptPath = plugin_script_path;
        plugin_describe_info_map_[current_plugin_name_].runStatus = RemoteDiagPluginRunStatusType::kExecuting;
        respMessage["DATA"] = "Start running";
        RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
        bool finish = false;
        std::ifstream ifs;
        uint16_t iNumber = plugin_describe_info_map_[current_plugin_name_].runTimeout / 100 + 1;
        for (uint i = 0; i < iNumber; i++) {
            if (stop_flag_) {
                break;
            }

            if (access((plugin_result_path + "/" + RUN_FINISH_FILE_NAME).c_str(), F_OK) == 0) {
                finish = true;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (finish) {
            plugin_describe_info_map_[current_plugin_name_].runStatus = RemoteDiagPluginRunStatusType::kSucceed;
            if (plugin_describe_info_map_[current_plugin_name_].isNeedCompressResult) {
                plugin_describe_info_map_[current_plugin_name_].runResultPath = plugin_result_path;
            }
            else {
                plugin_describe_info_map_[current_plugin_name_].runResultPath = plugin_result_path + "/" + plugin_describe_info_map_[current_plugin_name_].unCompressFile;
            }
        }
        else {
            if (stop_flag_) {
                plugin_describe_info_map_[current_plugin_name_].runStatus = RemoteDiagPluginRunStatusType::kTerminated;
            }
            else {
                plugin_describe_info_map_[current_plugin_name_].runStatus = RemoteDiagPluginRunStatusType::kTimeout;
            }
        }

        RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_DEFAULT);
        RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(pluginInfo.sa.c_str(), 0, 16)), false);
    }, pluginInfo).detach();
}

void
RemoteDiagDynamicPlugin::GetPluginRunResult(const RemoteDiagPluginRunResultInfo& resultInfo)
{
    DGR_INFO << "RemoteDiagDynamicPlugin::GetPluginRunResult pluginName: " << resultInfo.pluginName;
    RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_GET_PLUGIN_RUN_RESULT);
    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(resultInfo.sa.c_str(), 0, 16)), true);
    Json::Value respMessage;
    respMessage["SA"] = resultInfo.ta;
    respMessage["TA"] = resultInfo.sa;
    respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kPluginRunResult];

    std::string pluginName = resultInfo.pluginName;
    std::string plugin_path = plugin_package_dir_path_ + resultInfo.pluginName;
    if (("" == resultInfo.pluginName) || (current_plugin_name_ == resultInfo.pluginName) || (access(plugin_path.c_str(), F_OK) != 0)) {
        pluginName = current_plugin_name_;
    }

    auto itr_find = plugin_describe_info_map_.find(pluginName);
    if (itr_find == plugin_describe_info_map_.end()) {
        respMessage["RUN_STATUS"] = REMOTE_DIAG_PLUGIN_RUN_STATUS[RemoteDiagPluginRunStatusType::kDefault];
        respMessage["DATA"] = "The plugin has not been executed before.";
    }
    else {
        respMessage["RUN_STATUS"] = REMOTE_DIAG_PLUGIN_RUN_STATUS[itr_find->second.runStatus];
        if (itr_find->second.isNeedCompressResult) {
            respMessage["RESULT_TYPE"] = "DIR";
        }
        else {
            respMessage["RESULT_TYPE"] = "FILE";
        }

        respMessage["DATA"] = itr_find->second.runResultPath;
    }

    // update result status
    if (RemoteDiagPluginRunStatusType::kSucceed == itr_find->second.runStatus) {
        itr_find->second.runStatus = RemoteDiagPluginRunStatusType::kResult;
    }

    RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
    RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_DEFAULT);
    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(resultInfo.sa.c_str(), 0, 16)), false);
}

void
RemoteDiagDynamicPlugin::CheckPluginResult(const std::string& filePath)
{
    DGR_INFO << "RemoteDiagDynamicPlugin::CheckPluginResult filePath: " << filePath;
    for (auto& item : plugin_describe_info_map_) {
        if (RemoteDiagPluginRunStatusType::kResult == item.second.runStatus) {
            if (filePath == item.second.runResultPath) {
                RemovePluginDirectory(item.second.pluginPath);
                break;
            }
        }
    }
}

char*
RemoteDiagDynamicPlugin::GetJsonAll(const char *fname)
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

    str = (char*)malloc(filesize);
    memset(str, 0, filesize);

    rewind(fp);
    while ((fgets(txt, 1000, fp)) != NULL) {
        strcat(str, txt);
    }
    fclose(fp);

    return str;
}

void
RemoteDiagDynamicPlugin::ParsePluginDescribeJson(const std::string& pluginPath)
{
    std::string describePath = pluginPath + "/describe/describe.json";
    DGR_INFO << "RemoteDiagDynamicPlugin::ParsePluginDescribeJson describePath: " << describePath;
    char* jsonstr = GetJsonAll(describePath.c_str());
    if (nullptr == jsonstr) {
        DGR_ERROR << "RemoteDiagDynamicPlugin::ParsePluginDescribeJson error jsonstr is nullptr.";
        return;
    }

    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value  rootValue;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);

    if (!res || !errs.empty()) {
        if (jsonstr != NULL) {
            free(jsonstr);
        }

        return;
    }

    // load plugin describe info
    RemoteDiagPluginDescribeInfo describeInfo;
    if (rootValue["DPluginName"].isString()) {
        describeInfo.pluginName = static_cast<std::string>(rootValue["DPluginName"].asString());
    }

    if (rootValue["DpluginDes"].isString()) {
        describeInfo.pluginDescribe = static_cast<std::string>(rootValue["DpluginDes"].asString());
    }

    if (rootValue["RunParameters"].isString()) {
        describeInfo.pluginRunParameters = static_cast<std::string>(rootValue["RunParameters"].asString());
    }

    if (rootValue["NeedCompressResult"].isBool()) {
        describeInfo.isNeedCompressResult = static_cast<bool>(rootValue["NeedCompressResult"].asBool());
    }

    if (rootValue["UnCompressFile"].isString()) {
        describeInfo.unCompressFile = static_cast<std::string>(rootValue["UnCompressFile"].asString());
    }

    if (rootValue["RunTimeout"].isUInt()) {
        describeInfo.runTimeout = static_cast<uint16_t>(rootValue["RunTimeout"].asUInt());
    }

    describeInfo.pluginPath = pluginPath;
    describeInfo.scriptPath = "";
    describeInfo.runStatus = RemoteDiagPluginRunStatusType::kDefault;
    describeInfo.runResultPath = "";

    auto itr_find = plugin_describe_info_map_.find(current_plugin_name_);
    if (itr_find != plugin_describe_info_map_.end()) {
        plugin_describe_info_map_[current_plugin_name_] = describeInfo;
    }
    else {
        plugin_describe_info_map_.insert(std::make_pair(current_plugin_name_, describeInfo));
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }
}

bool
RemoteDiagDynamicPlugin::PluginPackageVerification(const std::string& pluginPackagePath)
{
    DGR_INFO << "RemoteDiagDynamicPlugin::PluginPackageVerification pluginPackagePath: " << pluginPackagePath;
    std::string command = "";
    auto zip_idx = pluginPackagePath.find("zip");
    auto tar_idx = pluginPackagePath.find("tar");
    if(zip_idx != string::npos) {
        command = "cd " + plugin_package_dir_path_ + "\n" + "unzip -o " + pluginPackagePath;
    }
    else {
        if(tar_idx != string::npos) {
            command = "cd " + plugin_package_dir_path_ + "\n" + "tar -xf " + pluginPackagePath;
        }
    }

    if ("" == command) {
        return false;
    }

    int ret = system(command.c_str());
    if (0 != ret) {
        DGR_WARN << "RemoteDiagDynamicPlugin::PluginPackageVerification pluginPackagePath: " << pluginPackagePath << " decompression failed.";
        return false;
    }

    std::remove(pluginPackagePath.c_str());
    std::string plugin_path = "";
    for (auto& item : pluginPackagePath) {
        if ('.' == item) {
            break;
        }

        plugin_path += item;
    }

    std::string plugin_start_path = plugin_path + "/script/start.sh";
    if (access((plugin_start_path).c_str(), F_OK) != 0) {
        DGR_ERROR << "RemoteDiagDynamicPlugin::PluginPackageVerification plugin start script not exist.";
        RemovePluginDirectory(plugin_path);
        return false;
    }

    std::string plugin_stop_path = plugin_path + "/script/stop.sh";
    if (access((plugin_stop_path).c_str(), F_OK) != 0) {
        DGR_ERROR << "RemoteDiagDynamicPlugin::PluginPackageVerification plugin stop script not exist.";
        RemovePluginDirectory(plugin_path);
        return false;
    }

    std::string plugin_describe_path = plugin_path + "/describe/describe.json";
    if (access((plugin_describe_path).c_str(), F_OK) != 0) {
        DGR_ERROR << "RemoteDiagDynamicPlugin::PluginPackageVerification plugin describe file not exist.";
        RemovePluginDirectory(plugin_path);
        return false;
    }

    ParsePluginDescribeJson(plugin_path);
    return true;
}

void
RemoteDiagDynamicPlugin::RemovePluginDirectory(const std::string& pluginDirPath)
{
    DGR_INFO << "RemoteDiagDynamicPlugin::RemovePluginDirectory pluginDirPath: " << pluginDirPath;
    if (access((pluginDirPath).c_str(), F_OK) != 0) {
        DGR_WARN << "RemoteDiagDynamicPlugin::RemovePluginDirectory pluginDirPath  not exist.";
        return;
    }

    DIR* directory = opendir(pluginDirPath.c_str());
    if (directory) {
        dirent* entry;
        while ((entry = readdir(directory)) != nullptr) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            std::string sub_path = std::string(pluginDirPath) + "/" + entry->d_name;
            if (entry->d_type == DT_DIR) {
                RemovePluginDirectory(sub_path.c_str()); // recursively delete subfolders
            }
            else {
                std::remove(sub_path.c_str());
            }
        }

        closedir(directory);
    }

    std::remove(pluginDirPath.c_str());
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon