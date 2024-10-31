#include <string>
#include <map>
#include <memory>
#include <thread>
#include "adf-lite/include/topic_manager.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

using namespace hozon::netaos::adf_lite;
namespace hozon {
namespace netaos {
namespace adf_lite {

void TopicManager::AddRecvInstance(const std::string topic, const std::string cm_topic, std::shared_ptr<DsRecv> recv_instance) {
    _total_topic_info[topic].cm_topic = cm_topic;
    _total_topic_info[topic].recv_instance = recv_instance;
};

// 将哪些executor的哪些trigger，会包含
void TopicManager::AddTrigger(const std::string topic, const std::string executor, std::string trigger) {
    std::unique_lock<std::mutex> topic_lk(_total_topic_info[topic].topic_mutex);
    std::map<std::string, ExecutorInfo>& executor_triggers = _total_topic_info[topic].executor_triggers;
    _total_topic_info[topic].current_status = true;
    _total_topic_info[topic].change = false;
    executor_triggers[executor].trigger_status.insert(std::make_pair(trigger, true));

    ADF_INTERNAL_LOG_INFO << "AddTrigger topic: " << topic << " trigger: " << trigger << " result is " <<_total_topic_info[topic].executor_triggers.size();
};

// 如果没有指定recv，则认为是不需要接收CmTopic，没有必要修改，因为不会使用到。
void TopicManager::ModifyTriggerStatus(const std::string topic, const std::string executor, std::string trigger, bool status) {
    if (_total_topic_info.count(topic) > 0) {
        if (_total_topic_info[topic].recv_instance == nullptr) {
            ADF_INTERNAL_LOG_INFO << "topic " << topic << " 's recv_instance is null";
            return;
        }
        
        {
            std::unique_lock<std::mutex> topic_lk(_total_topic_info[topic].topic_mutex);
            std::map<std::string, ExecutorInfo>& executor_triggers = _total_topic_info[topic].executor_triggers;
            if (executor_triggers[executor].trigger_status.count(trigger) > 0) {
                executor_triggers[executor].trigger_status[trigger] = status;
            }
        }
        PauseCmTopic(topic);
    }
};

void TopicManager::PauseCmTopic(const std::string topic) {
    // 如果接收状态是false，则设置为不接收
    bool recv_status = CheckReceiveStatus(topic);
    if (_total_topic_info[topic].change) {
        ADF_INTERNAL_LOG_INFO << "topic " << topic << " 's current_status is changed";
        std::unique_lock<std::mutex> topic_lk(_total_topic_info[topic].topic_mutex);
        _total_topic_info[topic].change = false;
        _total_topic_info[topic].current_status = recv_status;
        if (recv_status) {
            _total_topic_info[topic].recv_instance->ResumeReceive();
        } else {
            _total_topic_info[topic].recv_instance->PauseReceive();
        }
    } else {
        ADF_INTERNAL_LOG_INFO << "topic " << topic << " 's current_status is not changed";
    }
}

/* 检查是否所有的trigger接收topic的状态都变为了false，或者从全false变为有true的*/
bool TopicManager::CheckReceiveStatus(const std::string topic) {
    if (_total_topic_info.count(topic) > 0) {
        std::unique_lock<std::mutex> topic_lk(_total_topic_info[topic].topic_mutex);
        std::map<std::string, ExecutorInfo>& executor_triggers = _total_topic_info[topic].executor_triggers;
        for (auto it = executor_triggers.begin(); it != executor_triggers.end(); it++) {
            for (auto it2 = it->second.trigger_status.begin(); it2 !=it->second.trigger_status.end(); it2++) {
                // 查找到某个trigger需要接收topic数据
                if (it2->second) {
                    if (_total_topic_info[topic].current_status == false) {
                        ADF_INTERNAL_LOG_INFO << "topic " << topic << " CheckReceiveStatus changed, current_status change to true";
                        _total_topic_info[topic].change = true;
                    }
                    return true;    // 只要有一个是true的，就不能暂停
                }
            }
        }
    }
    // 没有任何trigger需要接收topic数据，但current_status是true，说明需要不接收topic数据
    if (_total_topic_info[topic].current_status) {
        ADF_INTERNAL_LOG_INFO << "topic " << topic << " CheckReceiveStatus changed, current_status change to false";
        _total_topic_info[topic].change = true;
    }
    return false;   // 所有的都是false时，就会暂停
}

// 将map中的值打印出来
void TopicManager::PrintTopic() {
    for (auto tmp = _total_topic_info.begin(); tmp != _total_topic_info.end(); tmp++) {
        ADF_INTERNAL_LOG_INFO << "== topic: " << tmp->first << " cmTopic: " << tmp->second.cm_topic << " size is: " << tmp->second.executor_triggers.size();
        std::map<std::string, ExecutorInfo>& executor_triggers = tmp->second.executor_triggers;
        for (auto it = executor_triggers.begin(); it != executor_triggers.end(); it++) {
            ADF_INTERNAL_LOG_INFO << "\tname is: " << it->first << " size is: " << it->second.trigger_status.size();
            for (auto it2 = it->second.trigger_status.begin(); it2 !=it->second.trigger_status.end(); it2++) {
                if (it2->second) {
                    ADF_INTERNAL_LOG_INFO << "\t\ttrigger is: " << it2->first << " true";
                } else {
                    ADF_INTERNAL_LOG_INFO << "\t\ttrigger is: " << it2->first << " false";
                }
            }
        }
    }
}

}
}
}

