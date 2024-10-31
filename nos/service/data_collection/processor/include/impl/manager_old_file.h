/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: log_collector.h
 * @Date: 2023/09/26
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_LOG_COLLECTOR_H__
#define NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_LOG_COLLECTOR_H__

#include "processor/include/processor.h"
#include "utils/include/path_utils.h"
#include "utils/include/time_utils.h"
#include <sys/stat.h>
#include <filesystem>
#include <regex>
#include <cstdlib>

struct ManagerOldFileSearchConfig{
    std::string fileNameRegex;
    int getMaxFileNum{0};
    std::vector<std::string> searchFolderPath;
    bool searchSubPath{false};
    bool delOldBeforeGet{false};
    bool getOldestFile{true};
};
YCS_ADD_STRUCT(ManagerOldFileSearchConfig, fileNameRegex,getMaxFileNum,searchFolderPath,searchSubPath,getOldestFile,delOldBeforeGet);

namespace hozon {
namespace netaos {
namespace dc {

class ManagerOldFiles : public Processor {
   public:
    ManagerOldFiles() {
        DC_SERVER_LOG_DEBUG<<"======ManagerOldFiles was created. for_debug =======";
    }
    ~ManagerOldFiles() override {
    };
    void onCondition(std::string type, char *data, Callback callback) override {}

    void configure(std::string type, YAML::Node &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        fileSearchCfg_ = node.as<ManagerOldFileSearchConfig>();
    };
    void configure(std::string type, DataTrans &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        nodeResults_ = node;
    };


    bool getSubPathList(std::string path, std::string filePattern, std::vector<std::string>& pathList, int maxSize) {
        if (!PathUtils::isDirExist(path)) {
            DC_SERVER_LOG_ERROR << "the folder not exists for getFiles:" + path;
            return false;
        }
        bool result = true;
        for (const auto & entry : std::filesystem::directory_iterator(path)) {
            std::regex  expression(filePattern);
            if (std::regex_match (entry.path().filename().string(), expression)) {
                if (maxSize > pathList.size()) {
                    pathList.emplace_back(entry.path());
                    DC_SERVER_LOG_INFO << "managerOldFilesGet:" << entry.path();
                } else {
                    break;
                }
            }
            continue;
        }
        return result;
    }
    void active() override {
        status_.store(TaskStatus::RUNNING,std::memory_order::memory_order_release);
        if (stop_) {
            status_= TaskStatus::ERROR;
            return;
        }
        if (fileSearchCfg_.delOldBeforeGet && PathUtils::isFileExist(FILE_MANAGER_PATH)){
            for (auto &path: fileSearchCfg_.searchFolderPath) {
                if (!PathUtils::isDirExist(path) || stop_) {
                    continue;
                }
                std::string managerScript = std::string(FILE_MANAGER_PATH) + " \""+path+"\"";
                system(managerScript.c_str());
            }
        }
        if (stop_) {
            status_= TaskStatus::ERROR;
            return;
        }
        for (auto &path: fileSearchCfg_.searchFolderPath) {
            if (!PathUtils::isDirExist(path) || stop_) {
                continue;
            }
            getSubPathList(path,fileSearchCfg_.fileNameRegex, results_, fileSearchCfg_.getMaxFileNum);
        }
        status_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
    };
    void deactive() override{
        stop_.store(true, std::memory_order::memory_order_release);
    };
    TaskStatus getStatus() override {
        return status_.load(std::memory_order::memory_order_acquire);
    };
    bool getTaskResult(const std::string &type, struct DataTrans &dataStruct) override {
        std::lock_guard<std::mutex> lg(mtx_);
        dataStruct.dataType = DataTransType::file;
        dataStruct.pathsList[compressedFiles].insert(results_.begin(),results_.end());
        dataStruct.mergeDataStruct(nodeResults_);
        return true;
    };
    void pause() override {}
   private:
    std::atomic<TaskStatus> status_{TaskStatus::INITIAL};
    std::atomic_bool stop_{false};
    std::mutex mtx_;
    DataTrans nodeResults_;
    std::vector<std::string> results_;
    ManagerOldFileSearchConfig fileSearchCfg_;
    std::string type_;
    const char *FILE_MANAGER_PATH= "/app/scripts/dc_file_mgr.sh";
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_LOG_COLLECTOR_H__
