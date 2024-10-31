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

#include "collection/include/collection.h"
#include "utils/include/path_utils.h"
#include "utils/include/time_utils.h"
#include <sys/stat.h>
#include <filesystem>
#include <regex>
#include <stdlib.h>
#include "phm/include/phm_client.h"
#include <memory>
#include <future>
#include <chrono>

struct LogCollectorLogSearchConfig{
    std::string type;
    std::string logNameRegex;
    std::string dateIndexOrPattern;
    std::string dateFormatStr;
    std::vector<std::string> searchFolderPath;
    int triggerTimeOffsetSec{0};
    bool searchSubPath{false};
    bool refreshLog{false};
};
YCS_ADD_STRUCT(LogCollectorLogSearchConfig,type, logNameRegex,dateIndexOrPattern, dateFormatStr,triggerTimeOffsetSec,searchFolderPath,searchSubPath,refreshLog);

namespace hozon {
namespace netaos {
namespace dc {

class LogCollector : public Collection {
   public:
    LogCollector() {
        DC_SERVER_LOG_DEBUG<<"======log collector was created. for_debug =======";
        std::cout<<"======log collector was created. for_debug ======="<<__FILE__<<":"<<__LINE__<<std::endl;
    }
    ~LogCollector() override {
    };
    void onCondition(std::string type, char *data, Callback callback) override {}

    void configure(std::string type, YAML::Node &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        logSearchCfg_ = node.as<LogCollectorLogSearchConfig>();
    };
    void configure(std::string type, DataTrans &node) override {
        status_.store(TaskStatus::CONFIGURED,std::memory_order::memory_order_release);
        std::lock_guard<std::mutex> lg(mtx_);
        nodeResults_ = node;
    };
    void filterRepeatAndTime(std::time_t neededLogTime) {
        //去重， 选择时间。
        std::sort(results_.begin(),results_.end(), strCmp());
        std::vector<std::string> newResults;
        const int prefixOffset = -5;
        std::tm tm;
        DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
        for (auto &logPath: results_) {
            std::filesystem::path path = logPath;
            std::regex regexPattern;
            size_t index;
            if (logSearchCfg_.dateIndexOrPattern.size()<2) {
                regexPattern=std::regex(logSearchCfg_.logNameRegex);
                index = logSearchCfg_.dateIndexOrPattern[0]-'0';
                if (index<=0 || index>9) {
                    DC_SERVER_LOG_ERROR<<"dateIndexOrPattern index is not correct:"<< logSearchCfg_.dateIndexOrPattern;
                    continue;
                }
            }else {
                regexPattern=std::regex(logSearchCfg_.dateIndexOrPattern);
                index = 0;
            }
            std::smatch matchResult;
            std::string fileName = path.filename();
            if (!std::regex_search(fileName, matchResult, regexPattern)) {
                DC_SERVER_LOG_ERROR<<"dateIndexOrPattern not match the date:"<< logSearchCfg_.dateIndexOrPattern;
                continue;
            }
            std::cout << "匹配成功:" << fileName << std::endl;
            std::string dateStr = matchResult[index].str();
            strptime(dateStr.c_str(), logSearchCfg_.dateFormatStr.c_str(), &tm);
            time_t logFileTime = mktime(&tm);
            if (logFileTime > neededLogTime) {
                continue;
            }
            if (newResults.empty()) {
                newResults.emplace_back(logPath);
                continue;
            }
            int dateIndex = logPath.rfind(dateStr);
            if (newResults.back().substr(0, dateIndex+prefixOffset) == logPath.substr(0,dateIndex+prefixOffset)) {
                // 存在时间更早的日志， 替换之。
                newResults.back()=logPath;
            } else {
                newResults.emplace_back(logPath);
            }
        }
        DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
        results_ = newResults;
    }

    void active() override {
        status_.store(TaskStatus::RUNNING,std::memory_order::memory_order_release);
        DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
        std::lock_guard<std::mutex> lg(mtx_);
        if (BasicTask::collectFlag.logCollect == false) {
            DC_SERVER_LOG_DEBUG<<"log collect flag is false";
            results_.clear();
            status_.store(TaskStatus::FINISHED,std::memory_order::memory_order_release);
            DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
            return;
        }
        if (logSearchCfg_.refreshLog) {
            refresh_log();
        }
        std::time_t neededLogTime = std::time(nullptr)+logSearchCfg_.triggerTimeOffsetSec;
        DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
        struct stat st;
        for (auto &path: logSearchCfg_.searchFolderPath) {
            stat(path.c_str(), &st);
            if (st.st_mode & S_IFREG) {
                results_.emplace_back(path);
            } else if (st.st_mode & S_IFDIR) {
                auto result  = PathUtils::getFiles(path, logSearchCfg_.logNameRegex, logSearchCfg_.searchSubPath, results_);
                if (!result) {
                    DC_SERVER_LOG_ERROR<<"get files error,path:"<<path;
                }
            } else {
                DC_SERVER_LOG_ERROR<<path<<" is not folder or file. skip.";
                continue;
            }
        }
        DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
        filterRepeatAndTime(neededLogTime);
        DC_SERVER_LOG_WARN<<__FILE__<<":"<<__LINE__;
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
    static bool refresh_log() {
        using namespace netaos::phm;
        static std::shared_ptr<PHMClient> spPHMClient(new PHMClient(), [](PHMClient* pPHMClient) {
            pPHMClient->Deinit();
            delete pPHMClient;
        });
        static bool res = []() -> bool {
            auto ret = spPHMClient->Init("");
            if (ret == 0) {
                DC_SERVER_LOG_INFO << "PHMClient Init ok";
                return true;
            }
            DC_SERVER_LOG_ERROR << "PHMClient Init error " << ret;
            return false;
        }();
        if (res) {
            std::promise<std::vector<std::string>> promise_res;
            std::future<std::vector<std::string>> future_res = promise_res.get_future();
            auto ret = spPHMClient->GetDataCollectionFile(
                [&promise_res](const auto& files) mutable { promise_res.set_value(files); });
            auto status = future_res.wait_for(std::chrono::seconds(2));
            DC_SERVER_LOG_INFO << "PHMClient wait_for completed";
            if (ret == 0 && status == std::future_status::ready) {
                auto files = future_res.get();
                DC_SERVER_LOG_INFO << "PHMClient GetDataCollectionFile return " << files.size() << " files";
                for (const auto& file : files) {
                    DC_SERVER_LOG_DEBUG << file;
                }
                DC_SERVER_LOG_INFO << "refresh_log completed";
                return true;
            }
            DC_SERVER_LOG_ERROR << "PHMClient GetDataCollectionFile error";
        }
        return false;
    }
   private:
    std::atomic<TaskStatus> status_{TaskStatus::INITIAL};
    std::atomic_bool stop_{false};
    std::mutex mtx_;
    DataTrans nodeResults_;
    std::vector<std::string> results_;
    LogCollectorLogSearchConfig logSearchCfg_;
    std::string type_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_COMMIT_SERVICE_DATA_COLLECTION_COLLECTION_INCLUDE_IMPL_LOG_COLLECTOR_H__
