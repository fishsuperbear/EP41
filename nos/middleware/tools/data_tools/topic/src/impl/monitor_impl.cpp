#include <impl/monitor_impl.h>
#include <monitor/cyber_topology_message.h>
#include <monitor/screen.h>
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "topic_manager.hpp"

namespace hozon {
namespace netaos {
namespace topic {

MonitorImpl::MonitorImpl(){};

MonitorImpl::~MonitorImpl() {
    if (!_isStop) {
        Stop();
    }
};

void MonitorImpl::SigResizeHandle() {
    Screen::Instance()->Resize();
};

// void MonitorImpl::Registercm_CMTypes() {
//     //regist event type
//     eprosima::fastdds::dds::TypeSupport event_type(std::make_shared<CmProtoBufPubSubType>());
//     registercm_protobufTypes();
//     event_type.get()->auto_fill_type_information(true);
//     event_type.get()->auto_fill_type_object(false);
//     if (!TopicManager::GetInstance().getDomainParticipant()) {
//         std::cout << "getDomainParticipant nullptr" << std::endl;
//     }
//     eprosima::fastdds::dds::TypeSupport::ReturnCode_t code = event_type.register_type(TopicManager::getInstance().getDomainParticipant());
//     if (eprosima::fastdds::dds::TypeSupport::ReturnCode_t::RETCODE_OK != code) {
//         std::cout << "register_type error" << std::endl;
//     }

//     //regist method type
//     registerproto_methodTypes();
//     eprosima::fastdds::dds::TypeSupport method_type(std::make_shared<ProtoMethodBasePubSubType>());
//     method_type.get()->auto_fill_type_information(true);
//     method_type.get()->auto_fill_type_object(false);

// if (!TopicManager::getInstance().getDomainParticipant()) {
//     std::cout << "getDomainParticipant nullptr" << std::endl;
// }
// code = method_type.register_type(TopicManager::getInstance().getDomainParticipant());
// if (eprosima::fastdds::dds::TypeSupport::ReturnCode_t::RETCODE_OK != code) {
//     std::cout << "register_type error" << std::endl;
// }

//regist someip type
//     eprosima::fastdds::dds::TypeSupport someip_type(std::make_shared<CmSomeipBufPubSubType>());
//     registercm_someipbufTypes();
//     someip_type.get()->auto_fill_type_information(true);
//     someip_type.get()->auto_fill_type_object(false);
//     if (!TopicManager::getInstance().getDomainParticipant()) {
//         std::cout << "getDomainParticipant nullptr" << std::endl;
//     }
//     code = someip_type.register_type(TopicManager::getInstance().getDomainParticipant());
//     if (eprosima::fastdds::dds::TypeSupport::ReturnCode_t::RETCODE_OK != code) {
//         std::cout << "register_type error" << std::endl;
//     }
// }

// void MonitorImpl::HandleNewTopic(CyberTopologyMessage& topology_msg) {
//     while (!_isStop) {
//         TopicInfo topicInfo;
//         {
//             std::unique_lock<std::mutex> lck(_newTopic_condition_mtx);
//             _newTopic_cv.wait(lck, [&]() { return ((_newTopicQueue.size() > 0) || (_isStop == true)); });
//             if (_isStop == true) {
//                 break;
//             }
//             topicInfo = _newTopicQueue.front();
//             _newTopicQueue.pop();
//         }
//         topology_msg.TopologyChanged(topicInfo);
//         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }
// }

void MonitorImpl::Start(MonitorOptions monitor_options) {
    {

        if (!monitor_options.method) {
            std::cout << "\033[33m"
                      << "method topics won't be showed. Can use -m to show them."
                      << "\033[0m" << std::endl;
        }
        std::string targe_topic = "";
        if (monitor_options.events.size() > 0) {
            targe_topic = monitor_options.events[0];
        }

        _topology_msg = std::make_shared<CyberTopologyMessage>(targe_topic, topic_manager_);
        _monitor_all = monitor_options.monitor_all;
        _method = monitor_options.method;
        _auto_subscribe = false;
        SubBase::Start(monitor_options.events);

        // std::thread handleNewTopic_t(&MonitorImpl::HandleNewTopic, this, std::ref(topology_msg));

        // auto topology_callback = [this](TopicInfo topicInfo) {
        //     std::unique_lock<std::mutex> lck(_newTopic_condition_mtx);
        //     _newTopicQueue.push(topicInfo);
        //     _newTopic_cv.notify_all();
        // };
        // TopicManager::GetInstance().RegistNewTopicCallback(topology_callback);
        // TopicManager::GetInstance().Init();
        // Registercm_CMTypes();

        Screen* s = Screen::Instance();
        s->SetCurrentRenderMessage(_topology_msg.get());
        s->Init();
        s->Run();
        _topology_msg = nullptr;
        _isStop = true;
        _newTopic_cv.notify_all();
        // handleNewTopic_t.join();
        //{}确保topology_msg在TopicManager reset前被释放
    }
    SubBase::Stop();
    return;
}

void MonitorImpl::Stop() {
    _isStop = true;
    Screen::Instance()->Stop();
}

void MonitorImpl::OnNewTopic(TopicInfo topic_info) {

    if ("" != topic_info.topicName) {
        if ("CmProtoBuf" != topic_info.typeName && "CmSomeipBuf" != topic_info.typeName && "ProtoMethodBase" != topic_info.typeName && topic_info.typeName.find("ZeroCopy") == std::string::npos) {
            TOPIC_LOG_DEBUG << "type: " << topic_info.typeName << " is unsupported!";
            return;
        }

        if (!_method) {
            //不显示method
            if (topic_info.topicName.find("/request/") == 0 || topic_info.topicName.find("/reply/") == 0) {
                return;
            }
        }

        _topology_msg->TopologyChanged(topic_info);
    }
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon