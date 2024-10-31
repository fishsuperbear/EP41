/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: log_collector.h
 * @Date: 2023/09/26
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_FIXED_FILES_COLLECTOR_H__
#define NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_FIXED_FILES_COLLECTOR_H__

#include <stdlib.h>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>

#include <regex>
#include "collection/include/collection.h"
#include "include/magic_enum.hpp"
#include "utils/include/path_utils.h"
#include "utils/include/time_utils.h"

namespace fs = std::filesystem;
struct FixedFilesConfig {
    std::string type;
    std::string move2NewFolder;
    std::vector<std::string> searchFolderPath;
    std::vector<std::string> fileList;
    std::string regexMatch{".{1,100}"};
    bool filterNodes{false};
    bool searchSubPath{false};
};

YCS_ADD_STRUCT(FixedFilesConfig, type,move2NewFolder, searchFolderPath, fileList, regexMatch, filterNodes, searchSubPath);

namespace hozon {
namespace netaos {
namespace dc {

class FixedFilesCollector : public Collection {
   public:
    FixedFilesCollector() {
        DC_SERVER_LOG_DEBUG<<"======FixedFilesCollector was created. for_debug =======";
        std::cout<<"======FixedFilesCollector was created. for_debug ======="<<__FILE__<<":"<<__LINE__<<std::endl;
    }
    ~FixedFilesCollector() override {
    };
    void onCondition(std::string type, char *data, Callback callback) override {}
    void addFilesToResult() {
        for (auto &filePath: fixedFilesCfg_.fileList) {
            if (stop_.load(std::memory_order::memory_order_acquire)) {
                return;
            }
            if (PathUtils::isFileExist(filePath)) {
                std::regex  expression(fixedFilesCfg_.regexMatch);
                if (std::regex_match (PathUtils::getFileName(filePath), expression)) {
                    results_.emplace_back(filePath);
                }
                results_.push_back(filePath);
            } else {
                DC_SERVER_LOG_WARN<<filePath<<" file was not exists but configured to collect.";
            }
        }
    }
    void addFilesInFoldersToResult() {
        if (!fixedFilesCfg_.searchSubPath) {
            results_.insert(results_.end(),fixedFilesCfg_.searchFolderPath.begin(),fixedFilesCfg_.searchFolderPath.end());
            return;
        }
        std::vector<std::string> files;
        for (auto & folderPath : fixedFilesCfg_.searchFolderPath) {
            if (stop_.load(std::memory_order::memory_order_acquire)) {
                return;
            }
            if (PathUtils::isDirExist(folderPath)) {
                auto getResult = PathUtils::getFiles(folderPath,fixedFilesCfg_.regexMatch,true,files);
                if (!getResult) {
                    DC_SERVER_LOG_WARN<<"getFiles error found with folder: "<< folderPath ;
                }
            } else {
                DC_SERVER_LOG_WARN<< folderPath <<" folder was not exists but configured to collect.";
            }
        }
        results_.insert(results_.end(),files.begin(),files.end());
    }
    void configure(std::string type, YAML::Node &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        fixedFilesCfg_ = node.as<FixedFilesConfig>();
    };
    void configure(std::string type, DataTrans &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        if (!type.empty() && type !="default") {
            filesType_ = magic_enum::enum_cast<FilesInListType>(type).value();
        }
        std::map<FilesInListType, std::set<std::string>> pathsList;
        nodePathList_ = node;
    };

    void active() override {
        status_.store(TaskStatus::RUNNING,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
//        if (fixedFilesCfg_.filterNodes) {
//            for (auto path:nodePathList_) {
//                if (stop_.load(std::memory_order::memory_order_acquire)) {
//                    return;
//                }
//                if (PathUtils::isFileExist(path)) {
//                    fixedFilesCfg_.fileList.emplace_back(path);
//                }
//                if (PathUtils::isDirExist(path)) {
//                    fixedFilesCfg_.searchFolderPath.emplace_back(path);
//                }
//            }
//        } else {
//            results_.insert(results_.end(), nodePathList_.begin(),nodePathList_.end());
//        }
        if ((fixedFilesCfg_.type == "faultManagerFiles") && (BasicTask::collectFlag.faultCollect == false)) {
            DC_SERVER_LOG_DEBUG<<"fault collect flag is false";
            results_.clear();
            status_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
            return;
        }
        if ((fixedFilesCfg_.type == "calibrationFiles") && (BasicTask::collectFlag.calibrationCollect == false)) {
            DC_SERVER_LOG_DEBUG<<"calibration collect flag is false";
            results_.clear();
            status_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
            return;
        }
        addFilesToResult();
        addFilesInFoldersToResult();
        std::string newFolder=fixedFilesCfg_.move2NewFolder;
        if (newFolder.size()>10) {
            PathUtils::createFoldersIfNotExists(newFolder);
            if (newFolder[newFolder.size()-1] != '/') {
                newFolder+="/";
            }
            std::vector<std::string> newResults;
            for (auto &item: results_) {
                std::string newFilePath = newFolder + fs::path(item).filename().string();
                fs::rename(item, newFilePath);
                newResults.emplace_back(newFilePath);
            }
            results_ = newResults;
        }
        status_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
    };
    void deactive() override{
        stop_.store(true, std::memory_order::memory_order_release);
    };
    TaskStatus getStatus() override {
        return status_.load(std::memory_order::memory_order_acquire);
    };
    bool getTaskResult(const std::string &type, struct DataTrans& dataStruct) override {
        std::lock_guard<std::mutex> lg(mtx_);
        dataStruct.dataType = DataTransType::file;
        dataStruct.pathsList[filesType_].insert(results_.begin(),results_.end());
        dataStruct.mergeDataStruct(nodePathList_);
        return true;
    };
    void pause() override {}
   private:
    std::atomic<TaskStatus> status_{TaskStatus::INITIAL};
    std::atomic_bool stop_{false};
    std::mutex mtx_;
    std::vector<std::string> results_;
    FilesInListType filesType_{faultManagerFiles};
    DataTrans nodePathList_;
    FixedFilesConfig fixedFilesCfg_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_FIXED_FILES_COLLECTOR_H__
