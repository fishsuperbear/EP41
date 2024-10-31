#pragma once

#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include "adf-lite/include/executor_mgr.h"

using namespace hozon::netaos::adf_lite;
namespace hozon {
namespace netaos {
namespace adf_lite {

struct TriggerStatus {
    std::string name;
    bool status;
};
class DbgInfo {
public:
    static DbgInfo& GetInstance() {
        static DbgInfo instance;

        return instance;
    };
    std::vector<std::string> GetExecutors();
    std::vector<std::string> GetExecutorTopics(const std::string executor_name);
    std::vector<std::string> GetTopicInfo(const std::string topic);
    std::unordered_map<std::string, std::vector<TriggerStatus>> GetAllInfo();
    void AddMgrMap(std::unordered_map<std::string, std::shared_ptr<ExecutorMgr>>& tt);
private:
    DbgInfo();

private:
    //std::map<std::string, ExecutorInfo> executors_info;
    //std::unordered_map<std::string, std::shared_ptr<ExecutorMgr>>& executor_mgr;
    // std::shared_ptr<hozon::netaos::adf_lite::TriggerControl> test;
    std::unordered_map<std::string, std::shared_ptr<ExecutorMgr>> * mgr_map;
};
}
}
}
