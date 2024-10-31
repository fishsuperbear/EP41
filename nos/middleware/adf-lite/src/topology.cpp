#include "adf-lite/include/topology.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

std::shared_ptr<DataQueue> 
Topology::CreateRecvQueue(const std::string& topic, uint32_t capacity) {
    std::lock_guard<std::mutex> lk(_mtx);
    if (routing_table_locked) {
        ADF_INTERNAL_LOG_ERROR << "Create recv queue for " << topic << " after routing table is locked.";
        return nullptr;
    }

    RoutingAttr attr;
    attr.topic = topic;
    attr.queue = std::make_shared<DataQueue>(capacity);

    _routing_table[topic].emplace_back(attr);
    ADF_INTERNAL_LOG_DEBUG << "Create recv queue for " << topic << ", capacity " << capacity;

    return attr.queue;
}

std::vector<RoutingAttr> Topology::AppendQueueForeachTopic(uint32_t capacity) {
    std::lock_guard<std::mutex> lk(_mtx);
    std::vector<RoutingAttr> queue_list;

    ADF_INTERNAL_LOG_DEBUG << "Append recv queue for each topic" << ", capacity " << capacity;
    for (auto& ele : _routing_table) {
        RoutingAttr attr;
        attr.topic = ele.first;
        attr.queue = std::make_shared<DataQueue>(capacity);

        _routing_table[attr.topic].emplace_back(attr);
        queue_list.emplace_back(attr);
    }

    return queue_list;
}

void Topology::GenRoutingTable() {
    std::lock_guard<std::mutex> lk(_mtx);
    routing_table_locked = true;
}

void Topology::Send(const std::string& topic, BaseDataTypePtr data) {
    if (_routing_table.find(topic) == _routing_table.end()) {
        ADF_INTERNAL_LOG_WARN << "Cannot find any receiver of " << topic;
        
        return;
    }

    auto& receivers = _routing_table[topic];
    for (auto& attr : receivers) {
        attr.queue->PushOneAndNotify(data);
        ADF_INTERNAL_LOG_VERBOSE << "Send to queue of " << topic << " receivers.size " << receivers.size();
    }
}

Topology::Topology() {

}

}
}
}