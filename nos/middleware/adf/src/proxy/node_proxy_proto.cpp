#include "adf/include/proxy/node_proxy_proto.h"
#include "adf/include/base.h"
#include "adf/include/data_types/common/types.h"
#include "adf/include/internal_log.h"
#include "idl/generated/cm_protobuf.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeProxyProto::NodeProxyProto(const NodeConfig::CommInstanceConfig& config,
                               std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type)
    : NodeProxyCM(config, pub_sub_type) {
    _freq_monitor.Start();
}

NodeProxyProto::~NodeProxyProto() {
    _freq_monitor.Stop();
}

BaseDataTypePtr NodeProxyProto::CreateBaseDataFromProto(std::shared_ptr<google::protobuf::Message> msg) {
    BaseDataTypePtr base_ptr(new BaseData);
    base_ptr->proto_msg = msg;
    ParseProtoHeader(msg, base_ptr->__header);

    return base_ptr;
}

void NodeProxyProto::OnDataReceive(void) {
    std::shared_ptr<google::protobuf::Message> proto_msg = ProtoMessageTypeMgr::GetInstance().Create(_config.name);
    if (proto_msg == nullptr) {
        ADF_LOG_ERROR << "Unknown protobuf type " << _config.name;
        return;
    }

    std::shared_ptr<CmProtoBuf> idl_msg(new CmProtoBuf);
    _proxy->Take(idl_msg);
    ADF_LOG_TRACE << "Proxy receive " << _config.name;

    bool ret = proto_msg->ParseFromArray(idl_msg->str().data(), idl_msg->str().size());
    if (!ret) {
        ADF_LOG_ERROR << "Fail to parse protobuf " << _config.name;
        return;
    }

    BaseDataTypePtr alg_data = CreateBaseDataFromProto(proto_msg);

    PushOneAndNotify(alg_data);
    _freq_monitor.PushOnce();
}

void NodeProxyProto::ParseProtoHeader(std::shared_ptr<google::protobuf::Message> proto_msg, Header& header) {
    const google::protobuf::Reflection* reflection = proto_msg->GetReflection();
    const google::protobuf::Descriptor* desc = proto_msg->GetDescriptor();
    const google::protobuf::FieldDescriptor* header_desc = desc->FindFieldByName("header");
    if (!header_desc) {
        ADF_LOG_INFO << "Missing header in " << proto_msg->GetTypeName();
        return;
    }
    const google::protobuf::Message& header_msg = reflection->GetMessage(*proto_msg, header_desc);
    header.seq = header_msg.GetReflection()->GetInt32(header_msg,
                                                      ::hozon::common::Header::GetDescriptor()->FindFieldByName("seq"));
    double timestamp_sec = header_msg.GetReflection()->GetDouble(
        header_msg, ::hozon::common::Header::GetDescriptor()->FindFieldByName("publish_stamp"));
    header.timestamp_real_us = TimestampToUs(timestamp_sec);
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
