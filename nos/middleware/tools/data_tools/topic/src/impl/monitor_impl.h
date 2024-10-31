
#pragma once
#include <monitor.h>
#include <condition_variable>
#include <memory>
#include <queue>
#include <monitor/cyber_topology_message.h>
#include "sub_base.h"

namespace hozon {
namespace netaos {
namespace topic {

class MonitorImpl : public SubBase {
   public:
    MonitorImpl();
    ~MonitorImpl();
    void Start(MonitorOptions monitor_options);
    void Stop();
    void SigResizeHandle();
    // void HandleNewTopic(CyberTopologyMessage& topology_msg);

   protected:
    // virtual void OnDataAvailable(eprosima::fastdds::dds::DataReader* reader) override;
    virtual void OnNewTopic(TopicInfo topic_info);

   private:
    // void Registercm_CMTypes();
    // bool _isStop = false;

    //new topic
    std::mutex _newTopic_condition_mtx;
    std::condition_variable _newTopic_cv;
    std::queue<TopicInfo> _newTopicQueue;
    std::shared_ptr<CyberTopologyMessage> _topology_msg;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon