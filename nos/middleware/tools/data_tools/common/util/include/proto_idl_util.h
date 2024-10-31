#include <fastdds/rtps/common/SerializedPayload.h>
#include <google/protobuf/message.h>

namespace hozon {
namespace netaos {
namespace bag {

class ProtoIdlUtil final {
   public:
    //将SerializedPayload_t转换成proto
    template <typename ProtoType>
    void SerializedPayloadToProto(const eprosima::fastrtps::rtps::SerializedPayload_t* data_payload, ProtoType& proto_data, std::string idl_type = "CmProtoBuf") {
        ConvertToProto(&proto_data, data_payload, proto_data.GetTypeName(), idl_type);
        return;
    }

    //将proto换成SerializedPayload_t
    template <typename ProtoType>
    void ProtoToSerializedPayload(const ProtoType& proto_data, eprosima::fastrtps::rtps::SerializedPayload_t* data_payload, std::string idl_type = "CmProtoBuf") {
        ConvertToProtoSerializedPayload(data_payload, &proto_data, proto_data.GetTypeName(), idl_type);
    }

    void ConvertToProto(google::protobuf::Message* proto_message, eprosima::fastrtps::rtps::SerializedPayload_t* data_payload, const std::string& proto_name, const std::string& idl_type);
    void ConvertToProtoSerializedPayload(eprosima::fastrtps::rtps::SerializedPayload_t* data_payload, const google::protobuf::Message* proto_message, const std::string& proto_name,
                                         const std::string& idl_type);
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon