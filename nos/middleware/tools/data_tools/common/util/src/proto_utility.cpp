#include "proto_utility.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {

bool ProtoUtility::HasProtoField(const google::protobuf::Message& message, std::string field_name) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    const google::protobuf::FieldDescriptor* field_descriptor = descriptor->FindFieldByName(field_name);
    return ((field_descriptor != nullptr) && reflection->HasField(message, field_descriptor));
}

int32_t ProtoUtility::GetProtoReflectionInt32(const google::protobuf::Message& message, std::string field_name) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    const google::protobuf::FieldDescriptor* field_descriptor = descriptor->FindFieldByName(field_name);
    if (field_descriptor && field_descriptor->type() == google::protobuf::FieldDescriptor::TYPE_INT32) {
        return reflection->GetInt32(message, field_descriptor);
    }
    return 0;
}

double ProtoUtility::GetProtoReflectionDouble(const google::protobuf::Message& message, std::string field_name) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    const google::protobuf::FieldDescriptor* field_descriptor = descriptor->FindFieldByName(field_name);
    if (field_descriptor && field_descriptor->type() == google::protobuf::FieldDescriptor::TYPE_DOUBLE) {
        return reflection->GetDouble(message, field_descriptor);
    }
    return 0.0;
}

const google::protobuf::Message& ProtoUtility::GetProtoReflectionMessage(const google::protobuf::Message& message, std::string field_name) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    const google::protobuf::FieldDescriptor* field_descriptor = descriptor->FindFieldByName(field_name);
    if (field_descriptor && field_descriptor->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
        return reflection->GetMessage(message, field_descriptor);
    }
    return *static_cast<const google::protobuf::Message*>(nullptr);
}

google::protobuf::Message& ProtoUtility::GetProtoReflectionMutableMessage(google::protobuf::Message& message, std::string field_name) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    const google::protobuf::FieldDescriptor* field_descriptor = descriptor->FindFieldByName(field_name);
    if (field_descriptor && field_descriptor->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
        return *(reflection->MutableMessage(&message, field_descriptor));
    }
    return *static_cast<google::protobuf::Message*>(nullptr);
}

bool ProtoUtility::SetProtoReflectionDouble(google::protobuf::Message& message, std::string field_name, double value) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    const google::protobuf::FieldDescriptor* field_descriptor = descriptor->FindFieldByName(field_name);
    if (field_descriptor && field_descriptor->type() == google::protobuf::FieldDescriptor::TYPE_DOUBLE) {
        reflection->SetDouble(&message, field_descriptor, value);
        return true;
    }
    return false;
}

}
}
}