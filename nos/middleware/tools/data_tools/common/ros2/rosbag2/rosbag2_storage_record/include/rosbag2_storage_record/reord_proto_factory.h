#pragma once
#include "cyber/proto/proto_desc.pb.h"
#include "google/protobuf/descriptor.pb.h"

namespace rosbag2_storage_record::internal {

// using namespace google::protobuf;

class RecordProtoFactory {
   public:
    ~RecordProtoFactory();

    static RecordProtoFactory& getInstance();

    // Serialize all descriptors of the given message to string.
    static void GetDescriptorString(const google::protobuf::Message& message, std::string* desc_str);

    // // Serialize all descriptors of the descriptor to string.
    static void GetDescriptorString(const google::protobuf::Descriptor* desc, std::string* desc_str);

   private:
    RecordProtoFactory();
    RecordProtoFactory(const RecordProtoFactory&) = delete;
    RecordProtoFactory& operator=(const RecordProtoFactory&) = delete;
    google::protobuf::Message* GetMessageByGeneratedType(const std::string& type) const;
    static bool GetProtoDesc(const google::protobuf::FileDescriptor* file_desc, apollo::cyber::proto::ProtoDesc* proto_desc);
};

}  // namespace rosbag2_storage_record::internal