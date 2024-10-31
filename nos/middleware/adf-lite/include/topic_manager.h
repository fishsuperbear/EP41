#pragma once

#include <string>
#include <map>
#include <memory>
#include <thread>
#include "adf-lite/ds/ds_recv/ds_recv.h"
#include "adf-lite/include/executor.h"

using namespace hozon::netaos::adf_lite;
namespace hozon {
namespace netaos {
namespace adf_lite {
struct ExecutorInfo {
    std::map<std::string, bool> trigger_status;
};

//保存某个内部Topic，有哪些executor的哪些trigger需要接收。以及是否在接收中的状态
struct TopicInfo {
    bool current_status; //记录topic当前状态，
    bool change = false;
    std::mutex topic_mutex;
    std::string cm_topic;
    std::shared_ptr<DsRecv> recv_instance = nullptr;
    //executor列表，元素是每个executor的triggers的map
    std::map<std::string, ExecutorInfo> executor_triggers;
};

/*某个Source(topic/CmTopic)有哪些executor的哪些trigger会接收
std::map<std::string, hozon::netaos::adf_lite::TopicInfo> _total_topic_info;数据结构如下
  topic1:
        current_status                          当前是否要接收topic1(实际用于是否要接收对应的cm_topic)
        change                                  为了避免每个修改trigger的status时都会调用CmTopic的PauseReceive/ResumeReceive
        topic_mutex                             防止多线程读写冲突
        cm_topic                                对应的cm_topic
        recv_instance                           对应的CmTopic的RecvInstance，根据executor的config配置，在ds_executor.cpp中g_ds_recv_map中查找
        executor_name:                          executor的名称
            trigger1:status                     executor的trigger1，及是否接收topic1的状态
            trigger2:status                     executor的trigger2，及是否接收topic1的状态

主要思路：
指定了要暂停某个trigger时，首先确定该trigger的所有source，然后要停掉这些source，但如果有其它trigger(包括其它的executor的trigger)仍需要接收时，则不停。
不仅内部topic要停，如果是cmTopic传进来的cmTopic也要停。
内部的好控制，只要设置pause_enable就可以了。
对于cmTopic。
从map中，根据topic，取得对应的cmTopic，然后设置有哪些executor的哪些trigger在接收。
全设置完后，判断是不是所有接收的都是pause状态，如果是的话，就会将topic暂停。

要恢复时，如果某个trigger要接收，就会开启，

对于不来自cmTopic的topic,也会设置trigger的状态，但内部topic只需要设置pause_enable就可以了，所以没什么用。

注意：
map中没有保存CmTopic的domainId，因为实际中不存在多个CmTopic对应一个topic的场景，故一个topic最多对应一个RecvInstance。
*****************************/
class TopicManager {
public:
    static TopicManager& GetInstance() {
        static TopicManager instance;
        return instance;
    }

    // 对于CmTopic，追加接收该CmTopic的Instance
    void AddRecvInstance(const std::string topic, const std::string cm_topic, std::shared_ptr<DsRecv> recv_instance);

    // 将哪些executor的哪些trigger，会包含
    void AddTrigger(const std::string topic, const std::string executor, std::string trigger);

    // 如果没有指定recv，则认为是不需要接收CmTopic，没有必要修改，因为不会使用到。
    void ModifyTriggerStatus(const std::string topic, const std::string executor, std::string trigger, bool status);

    /* 检查是否所有的trigger接收topic的状态都变为了false，或者从全false变为有true的*/
    bool CheckReceiveStatus(const std::string topic);

    // 将map中的值打印出来
    void PrintTopic();

private:
    void PauseCmTopic(const std::string topic);
    std::map<std::string, hozon::netaos::adf_lite::TopicInfo> _total_topic_info;
};

}
}
}

