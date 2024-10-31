#pragma once

#include <google/protobuf/message.h>

namespace hozon {
namespace netaos {
namespace data_tool_common {

class ProtoUtility {
public:
    static bool HasProtoField(const google::protobuf::Message& message, std::string field_name);
    static int32_t GetProtoReflectionInt32(const google::protobuf::Message& message, std::string field_name);
    static double GetProtoReflectionDouble(const google::protobuf::Message& message, std::string field_name);
    static const google::protobuf::Message& GetProtoReflectionMessage(const google::protobuf::Message& message, std::string field_name);
    static google::protobuf::Message& GetProtoReflectionMutableMessage(google::protobuf::Message& message, std::string field_name);
    static bool SetProtoReflectionDouble(google::protobuf::Message& message, std::string field_name, double value);
};

}
}
}