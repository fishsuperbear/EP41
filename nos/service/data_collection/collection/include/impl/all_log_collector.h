/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: log_collector.h
 * @Date: 2023/09/26
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_ALL_LOG_COLLECTOR_H__
#define NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_ALL_LOG_COLLECTOR_H__

#include "collection/include/collection.h"
#include "utils/include/path_utils.h"
#include "utils/include/time_utils.h"
#include <sys/stat.h>
#include <filesystem>
#include <regex>
#include <stdlib.h>
#include <ctime>
#include "phm/include/phm_client.h"
#include <memory>
#include <queue>

struct collectLogPathConfig{
    std::string path;
    int sizeMb;
    std::string mode;
};
YCS_ADD_STRUCT(collectLogPathConfig,path, sizeMb,mode);

namespace hozon {
namespace netaos {
namespace dc {

struct fileInfo{
    std::string fileFullPath;
    int fileSizeKb;
    std::time_t modifyTime;
};

struct timeCmp{
    bool operator() (fileInfo &a, fileInfo &b) {
        return a.modifyTime < b.modifyTime;
    }
};

class AllLogCollector : public Collection {
   public:
    AllLogCollector() {
        DC_SERVER_LOG_DEBUG<<"======log collector was created. for_debug =======";
        std::cout<<"======log collector was created. for_debug ======="<<__FILE__<<":"<<__LINE__<<std::endl;
    }
    ~AllLogCollector() override {
    };
    void onCondition(std::string type, char *data, Callback callback) override {}

    void configure(std::string type, YAML::Node &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        folderList_ = node["folderList"].as<std::vector<collectLogPathConfig>>();
    };
    void configure(std::string type, DataTrans &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        nodeResults_ = node;
    };
    void active() override {
        status_.store(TaskStatus::RUNNING,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        for (auto &folderInfo: folderList_) {
            if (stop_.load(std::memory_order::memory_order_acquire)) {
                break;
            }
            getFilesNotExceedSize(folderInfo.path, folderInfo.sizeMb, results_);
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
        dataStruct.pathsList[hzLogFiles].insert(results_.begin(),results_.end());
        dataStruct.mergeDataStruct(nodeResults_);
        return true;
    };
    void pause() override {}

    bool getFilesNotExceedSize(const std::string& path, int sizeMb, std::vector<std::string>& files) {
        if (!PathUtils::isDirExist(path)) {
            DC_SERVER_LOG_ERROR << "the folder not exists for getFiles:" + path;
            return false;
        }
        std::priority_queue<fileInfo, std::vector<fileInfo>,timeCmp> fileQueue;
        getFiles(path, fileQueue);
        int sizeKb = sizeMb * 1000;
        while (sizeKb >0 && !fileQueue.empty()) {
            if (stop_.load(std::memory_order::memory_order_acquire)) {
                break;
            }
            files.push_back(fileQueue.top().fileFullPath);
            sizeKb -= fileQueue.top().fileSizeKb;
            fileQueue.pop();
        }
        DC_SERVER_LOG_INFO<<path<<" has "<<sizeKb/1000<<" Mb quota left.";
        return true;
    }

    static time_t getModifyTime(const std::filesystem::directory_entry& entry) {
        std::smatch matchResult;
        std::tm tm{};
        std::string timeFormat = "20\\d{2}-[01]\\d-[0-3]\\d_[0-6]\\d-[0-6]\\d-[0-6]\\d";
        std::regex regexPattern=std::regex(timeFormat);
        std::string fileName = entry.path().filename();
        if (std::regex_search(fileName, matchResult, regexPattern)) {
            std::string dateStr = matchResult[0].str();
            strptime(dateStr.c_str(), "%Y-%m-%d_%H-%M-%S", &tm);
            return mktime(&tm);
        }
        struct stat fileStat{};
        if (stat(entry.path().c_str(), &fileStat) == 0) {
            return fileStat.st_mtime;
        } else {
            DC_SERVER_LOG_ERROR << "Error getting file status." ;
            return 0;
        }
    }

    bool getFiles(const std::string& path, std::priority_queue<fileInfo, std::vector<fileInfo>,timeCmp>& files) {
        if (!PathUtils::isDirExist(path)) {
            DC_SERVER_LOG_ERROR << "the folder not exists for getFiles:" + path;
            return false;
        }
        bool result = true;
        for (const auto & entry : std::filesystem::directory_iterator(path)) {
            if (stop_.load(std::memory_order::memory_order_acquire)) {
                break;
            }
            if (std::filesystem::is_regular_file(entry)) {
                fileInfo fi;
                fi.modifyTime = getModifyTime(entry);
                fi.fileFullPath = entry.path();
                fi.fileSizeKb = std::filesystem::file_size(entry.path())/1000;
                files.push(fi);
                continue;
            }
            if (!getFiles(entry.path().string(), files)) {
                result = false;
            }
        }
        return result;
    }
   private:
    std::atomic<TaskStatus> status_{TaskStatus::INITIAL};
    std::atomic_bool stop_{false};
    std::mutex mtx_;
    DataTrans nodeResults_;
    std::vector<std::string> results_;
    std::vector<collectLogPathConfig> folderList_;
    std::string type_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_ALL_LOG_COLLECTOR_H__
