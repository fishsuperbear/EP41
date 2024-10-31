#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "adf-lite/include/cv_queue.h"
#include "adf-lite/include/base.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

using DataQueue = CVSizeLimitQueueWithFreq<BaseDataTypePtr>;

struct RoutingAttr {
    std::string topic;
    std::shared_ptr<DataQueue> queue;
};

class Topology {
public:
    static Topology& GetInstance() {
        static Topology instance;

        return instance;
    }

    std::shared_ptr<DataQueue> CreateRecvQueue(const std::string& topic, uint32_t capacity);
    std::vector<RoutingAttr> AppendQueueForeachTopic(uint32_t capacity);
    void GenRoutingTable();
    void Send(const std::string& topic, BaseDataTypePtr data);
    
private:
    Topology();
    std::unordered_map<std::string, std::vector<RoutingAttr>> _routing_table;
    std::mutex _mtx;
    bool routing_table_locked = false;
};

}
}
}