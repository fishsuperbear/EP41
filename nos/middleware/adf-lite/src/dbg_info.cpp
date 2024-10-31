#include <vector>
#include <unordered_map>
#include "adf-lite/include/dbg_info.h"
#include "adf-lite/include/topology.h"
#include "adf-lite/include/adf_lite_internal_logger.h"
#include "algorithm"
namespace hozon {
namespace netaos {
namespace adf_lite {

DbgInfo::DbgInfo() {

}

std::vector<std::string> DbgInfo::GetExecutors()
{
    std::cout << "=========GetExecutors===========" << std::endl;
    std::vector<std::string> result;
    for (auto& it:  *mgr_map)
    {
        std::cout << it.first << std::endl;
        std::cout << it.second->_config_file_path << std::endl;
        result.push_back(it.second->_config.executor_name);
    }
    std::cout << "============== size is " << result.size() << " =============================" << std::endl;
    return result;
}

std::vector<std::string> DbgInfo::GetExecutorTopics(const std::string executor_name)
{
    std::cout << "=========GetExecutorTopics===========" << std::endl;
    std::vector<std::string> result;
    for (auto& it:  *mgr_map)
    {
        if (it.second->_config.executor_name == executor_name) {
            for (auto &input: it.second->_config.inputs) {
                result.push_back(input.topic);
            }
            //for (auto &trigger: it.second->_trigger_control_map) {
            //    result.push_back(trigger.first);
            //}
            std::cout << "============== executor_name: " << executor_name << " found ===============" << std::endl;
            return result;
        }
    }
    std::cout << "============== executor_name: " << executor_name << " not found ===============" << std::endl;
    return result;
}

std::vector<std::string> DbgInfo::GetTopicInfo(const std::string topic)
{
    std::cout << "=========GetTopicInfo===========" << std::endl;
    std::vector<std::string> result;
    for (auto& it:  *mgr_map)
    {
        for (auto& it2: it.second->_config.inputs) {
            if (it2.topic == topic) {
                result.push_back(it.second->_config.executor_name);
            }
        }
        for (auto &trigger: it.second->_config.triggers) {
            for (auto &source: trigger.main_sources) {
                if (source.name == topic) {
                    result.push_back(it.second->_config.executor_name + "." + trigger.name);
                }
            }
        }
    }
    return result;
}


std::unordered_map<std::string, std::vector<TriggerStatus>> DbgInfo::GetAllInfo()
{
    std::unordered_map<std::string, std::vector<TriggerStatus>> result;
    std::cout << "=========GetAllInfo===========" << std::endl;
    for (auto& it:  *mgr_map)
    {
        std::vector<TriggerStatus> res;
        for (auto &trigger: it.second->_trigger_control_map) {
            TriggerStatus temp{trigger.first, trigger.second.pause_enable};
            res.push_back(temp);
        }
        result[it.second->_config.executor_name] = res;        
    }
    return result;
}

void DbgInfo::AddMgrMap(std::unordered_map<std::string, std::shared_ptr<ExecutorMgr>>& tt)
{
    mgr_map = &tt;
    // executors_info[executor_info.executor_name] = executor_info;
}

// void DbgInfo::AddExecutor2(std::unordered_map<std::string, std::shared_ptr<ExecutorMgr>>& mgr)
// {
//     executor_mgr = mgr;
// }

    
}
}
}