#include "proto_idl_util.h"
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "proto_factory.h"

namespace hozon {
namespace netaos {
namespace bag {

void ProtoIdlUtil::ConvertToProto(google::protobuf::Message* proto_message, eprosima::fastrtps::rtps::SerializedPayload_t* data_payload, const std::string& proto_name, const std::string& idl_type) {
    if ("CmProtoBuf" == idl_type) {
        CmProtoBuf cm_buf;
        CmProtoBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(data_payload, &cm_buf);
        if (proto_name == cm_buf.name()) {
            // proto_message = cm_buf.str();
            proto_message->ParseFromArray(cm_buf.str().data(), cm_buf.str().size());
        }

    } else if ("ProtoMethodBase" == idl_type) {
        ProtoMethodBase method_buf;
        ProtoMethodBasePubSubType method_pub_sub;
        method_pub_sub.deserialize(data_payload, &method_buf);
        if (proto_name == method_buf.name()) {
            proto_message->ParseFromArray(method_buf.str().data(), method_buf.str().size());
        }
    } else if ("CmSomeipBuf" == idl_type) {
        CmSomeipBuf cm_buf;
        CmSomeipBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(data_payload, &cm_buf);
        if (proto_name == cm_buf.name()) {
            proto_message->ParseFromArray(cm_buf.str().data(), cm_buf.str().size());
        }
    }
    return;
};

void ProtoIdlUtil::ConvertToProtoSerializedPayload(eprosima::fastrtps::rtps::SerializedPayload_t* data_payload, const google::protobuf::Message* proto_message, const std::string& proto_name,
                                                   const std::string& idl_type) {
    if ("CmProtoBuf" == idl_type) {
        CmProtoBuf cm_buf;
        CmProtoBufPubSubType sub_type;
        // Proto2cmPrtoBuf(serialized_string, proto_name, cm_buf);
        //获取proto对象序列化数据
        std::string proto_ser_data;
        if (!proto_message->SerializeToString(&proto_ser_data)) {
            BAG_LOG_ERROR << proto_message->GetTypeName() << " serialize to string failed!";
        }
        cm_buf.name(proto_message->GetTypeName());
        cm_buf.str().assign(proto_ser_data.begin(), proto_ser_data.end());
        data_payload->reserve(sub_type.getSerializedSizeProvider(&cm_buf)());
        sub_type.serialize(&cm_buf, data_payload);

    } else if ("ProtoMethodBase" == idl_type) {
        BAG_LOG_ERROR << idl_type << " unsupport!";
    } else if ("CmSomeipBuf" == idl_type) {
        BAG_LOG_ERROR << idl_type << " unsupport!";
    } else {
        BAG_LOG_ERROR << idl_type << " unsupport!";
    }
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon