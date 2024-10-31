#include "adf-lite/include/writer_impl.h"
#include "adf-lite/include/topology.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

WriterImpl::WriterImpl(Writer* writer) :
        _writer(writer) {
    (void)_writer;
}

WriterImpl::~WriterImpl() {

}

int32_t WriterImpl::Init(const std::string& topic) {
    _topic = topic;
    return 0;
}

int32_t WriterImpl::Write(BaseDataTypePtr data) {
    ParseProtoHeader(data);
    ADF_INTERNAL_LOG_VERBOSE << "Topology send " << _topic;
    Topology::GetInstance().Send(_topic, data);

    return 0;
}

void WriterImpl::ParseProtoHeader(BaseDataTypePtr& data) {
    if (data == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "data is nullptr";
        return;
    }
    // 如果是传递路径上的点，data->__header.timestamp_real_us中应该有值。如果是最初的源数据，值为0
    if (data->__header.timestamp_real_us != 0) {
        ADF_INTERNAL_LOG_DEBUG << "data header has timestamp, do nothing, value: " << data->__header.timestamp_real_us;
        return;
    }

    std::shared_ptr<google::protobuf::Message> proto_msg = data->proto_msg;

    if (proto_msg == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "proto_msg is nullptr";
        return;
    }
    const google::protobuf::Reflection* reflection = proto_msg->GetReflection();
    if (reflection == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "proto_msg reflection is nullptr";
        return;
    }
    const google::protobuf::Descriptor* desc = proto_msg->GetDescriptor();
    if (desc == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "proto_msg desc is nullptr";
        return;
    }

    const google::protobuf::FieldDescriptor* header_desc = desc->FindFieldByName("header");
    if (header_desc == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "Missing header in " << proto_msg->GetTypeName();
        return;
    }
    const google::protobuf::Message& header_msg = reflection->GetMessage(*proto_msg, header_desc);
    if (header_msg.GetReflection() == nullptr || ::hozon::common::Header::GetDescriptor() == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "header_msg reflection";
        return;
    }
    if (::hozon::common::Header::GetDescriptor()->FindFieldByName("seq") == nullptr ||
            ::hozon::common::Header::GetDescriptor()->FindFieldByName("publish_stamp") == nullptr) {
        ADF_INTERNAL_LOG_DEBUG << "header hasn't the member of seq and publish_stamp";
        return;
    }

    data->__header.seq = header_msg.GetReflection()->GetInt32(header_msg, ::hozon::common::Header::GetDescriptor()->FindFieldByName("seq"));

    double timestamp_sec
        = header_msg.GetReflection()->GetDouble(header_msg, ::hozon::common::Header::GetDescriptor()->FindFieldByName("publish_stamp"));
    data->__header.timestamp_real_us = TimestampToUs(timestamp_sec);
    ADF_INTERNAL_LOG_DEBUG << "_topic:" << _topic << " ParseProtoHeader timestamp_real_us is "  << data->__header.timestamp_real_us;
}

}
}
}