
#include "impl/list_impl.h"
#include <unordered_map>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include "cm/include/cm_config.h"
#include "data_tools_logger.hpp"

namespace hozon {
namespace netaos {
namespace topic {

ListImpl::ListImpl() {}

ListImpl::~ListImpl() {
    if (!_isStop) {
        Stop();
    }
}

void ListImpl::Stop() {
    _isStop = true;
    SubBase::Stop();
}

void ListImpl::Start(ListOptions list_options) {

    if (!list_options.method) {
        CONCLE_BAG_LOG_WARN << "method topics won't be showed. Can use -m to show them.";
    }
    _monitor_all = true;
    _method = list_options.method;
    _auto_subscribe = false;

    TOPIC_LOG_DEBUG << "topic hz start.";
    SubBase::Start({});

    //等待2秒，再打印
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "topic:" << std::endl;
    auto topics = topic_manager_->GetTopicInfo();
    for (auto item : topics) {
        if (!_method) {
            //不显示method
            if (item.first.find("/request/") == 0 || item.first.find("/reply/") == 0) {
                continue;
            }
        }
        std::cout << item.first << std::endl;
    }

    Stop();
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon
