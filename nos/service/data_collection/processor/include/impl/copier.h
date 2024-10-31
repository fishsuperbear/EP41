/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: copier.h
 * @Date: 2023/09/07
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_COPIER_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_COPIER_H__

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include "processor/include/processor.h"
#include "utils/include/path_utils.h"
#include "yaml-cpp/yaml.h"

namespace hozon::netaos::dc {


class Copier : public Processor {
   public:
    void onCondition(std::string type, char* data, Callback callback) override {}

    void configure(std::string type, YAML::Node& node) override {
        std::vector<std::string> pathList = node["pathList"].as<std::vector<std::string> >();
        outFolderUpper_ = node["outputFolder"].as<std::string>();
        std::lock_guard<std::mutex> lg(mtx_);
        pathList_.insert(pathList.begin(), pathList.end());
        if (!type.empty()) {
            filesType_ = magic_enum::enum_cast<FilesInListType>(type).value();
        }
        taskStatus_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
    }

    void configure(std::string type, DataTrans& node) override {
        std::lock_guard<std::mutex> lg(mtx_);
        nodePathList_ = node;
        taskStatus_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
    }

    void active() override {
        std::lock_guard<std::mutex> lg(mtx_);
        callTimes_++;
        if (outFolder_.empty()) {
            outFolder_ = PathUtils::getFilePath(outFolderUpper_, (TimeUtils::formatTimeStrForFileName(TimeUtils::getDataTimestamp())+"/"));
            PathUtils::createFoldersIfNotExists(outFolder_);
        }
        if (callTimes_ >6 || copiedFileList_.size()>=20 || stop_.load(std::memory_order::memory_order_acquire)) {
            taskStatus_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
            return ;
        }
        std::vector<std::string> allFiles;
        taskStatus_.store(TaskStatus::RUNNING,std::memory_order::memory_order_release);
        for (auto &folder: pathList_) {
            std::vector<std::string> tempFiles;

            for (const auto & entry : std::filesystem::directory_iterator(folder)) {
                if (std::filesystem::is_regular_file(entry)) {
                    tempFiles.emplace_back(entry.path());
                }
            }
            std::sort(tempFiles.begin(),tempFiles.end(), strCmp());
            auto tempLen = tempFiles.size();
            if (tempLen>2) {
                allFiles.insert(allFiles.end(),tempFiles.begin()+1,tempFiles.end()-1);
            }
            tempFiles.clear();
        }
        for (auto &filePath: allFiles) {
            if (std::find(copiedFileList_.begin(), copiedFileList_.end(), filePath)!= copiedFileList_.end()) {
                continue ;
            }
            std::filesystem::path  p(filePath);
            std::ifstream  src(filePath, std::ios::binary);

            std::string dstFilePath = outFolder_ +p.filename().c_str();
            std::ofstream  dst(dstFilePath,   std::ios::binary);
            dst << src.rdbuf();
            copiedFileList_.insert(filePath);
            copiedFileFullPathList_.emplace_back(dstFilePath);
        }
    }

    void deactive() override {
        stop_.store(true, std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        taskStatus_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
    }

    TaskStatus getStatus() override { return taskStatus_.load(std::memory_order::memory_order_acquire); };

    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override {
        if (taskStatus_.load(std::memory_order::memory_order_acquire)!=TaskStatus::FINISHED) {
            return false;
        }
        dataStruct.dataType = DataTransType::file;
        dataStruct.pathsList[filesType_].insert(copiedFileFullPathList_.begin(),copiedFileFullPathList_.end());
        dataStruct.mergeDataStruct(nodePathList_);
        return true;
    };

    void pause() override {}

    void doWhenDestroy(const Callback& callback) override { cb_ = callback; }

   private:
    Callback cb_;
    std::mutex mtx_;
    DataTrans nodePathList_;
    std::set<std::string> pathList_;
    std::set<std::string> copiedFileList_;
    FilesInListType filesType_{commonTopicMcapFiles};
    std::string outFolderUpper_;
    std::string outFolder_;
    std::vector<std::string> copiedFileFullPathList_;
    std::atomic<TaskStatus> taskStatus_;
    std::atomic_int callTimes_ = 0;
    std::atomic_bool stop_ = false;
};

}  // namespace hozon::netaos::dc

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_COPIER_H__
