#include "adf-lite/ds/ds_send/proto_cm_ds_send.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "adf/include/node_proto_register.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
ProtoCMDsSend::ProtoCMDsSend(const DSConfig::DataSource& config) :
        DsSend(config),
        _stop(false),
        _skeleton(std::make_shared<CmProtoBufPubSubType>()) {
    _reader.Init(_config.topic, _config.capacity);
    ResumeSend();
}

ProtoCMDsSend::~ProtoCMDsSend() {
}

void ProtoCMDsSend::PreDeinit() {
    DS_LOG_DEBUG << "Prepare Deinit: stop recv data thread";
    _stop = true;
}

void ProtoCMDsSend::Deinit() {
    DS_LOG_DEBUG << "Deinit: PauseSend()";
    PauseSend();
}

void ProtoCMDsSend::PauseSend() {
    _stop = true;
    if (_recv != nullptr) {
        _recv->join();
    }
    
    _skeleton.Deinit();
}

int32_t ProtoCMDsSend::Write(std::shared_ptr<google::protobuf::Message> data, Header& header) {
    if (data == nullptr) {
        DS_LOG_ERROR << "data pointer is nullptr!!!";
        return -1;
    }
    std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);
    if (cm_pb == nullptr) {
        DS_LOG_ERROR << "cm_pb pointer is nullptr!!!";
        return -1;
    }
    cm_pb->name(data->GetTypeName());

    std::string serialized_data;
    data->SerializeToString(&serialized_data);
    cm_pb->str().assign(serialized_data.begin(), serialized_data.end());

    DS_LOG_DEBUG << "header.latency_info.data size is: " << header.latency_info.data.size();
    for (auto data : header.latency_info.data) {
        LinkInfo link_info;
        link_info.link_name(data.first);
        link_info.timestamp_real().sec(data.second.sec);
        link_info.timestamp_real().nsec(data.second.nsec);
        DS_LOG_DEBUG << "Send Data copy latency_info: " << data.first
                     << " timestamp_real sec: " << link_info.timestamp_real().sec()
                     << " nsec: " << link_info.timestamp_real().nsec();
        cm_pb->header().latency_info().link_infos().emplace_back(link_info);
    }

    return _skeleton.Write(cm_pb);
}

void ProtoCMDsSend::ReceiveInnerTopicSendCmTopic() {
    while(!_stop) {
        hozon::netaos::adf_lite::BaseDataTypePtr ptr = _reader.GetLatestOneBlocking(1000, true);
        
        if (ptr != nullptr) {
            DS_LOG_DEBUG << "Receive Inner topic:" << _config.topic << "; Send CmTopic:" << _config.cm_topic;
            Write(ptr->proto_msg, ptr->__header);
        } else {
            DS_LOG_DEBUG << "GetLatestOne Timeout, topic:" << _config.topic;
        }
    }
}

void ProtoCMDsSend::ResumeSend() {
    _stop = false;
    _skeleton.Init(_config.cm_domain_id, _config.cm_topic);
    _recv = std::make_shared<std::thread>(std::bind(&ProtoCMDsSend::ReceiveInnerTopicSendCmTopic, this));
}

}
}
}

