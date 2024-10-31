#include "rosbag2_storage_record/reord_proto_factory.h"
#include <dirent.h>
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include "ament_index_cpp/get_search_paths.hpp"
// #include "proto_factory.h"
#include "data_tools_logger.hpp"

namespace rosbag2_storage_record::internal {

RecordProtoFactory::RecordProtoFactory() {}

RecordProtoFactory::~RecordProtoFactory() {}

RecordProtoFactory& RecordProtoFactory::getInstance() {
    static RecordProtoFactory instance;
    return instance;
}

void RecordProtoFactory::GetDescriptorString(const google::protobuf::Descriptor* desc, std::string* desc_str) {
    apollo::cyber::proto::ProtoDesc proto_desc;
    if (!GetProtoDesc(desc->file(), &proto_desc)) {
        COMMON_LOG_ERROR << "Failed to get descriptor from message";
        return;
    }

    if (!proto_desc.SerializeToString(desc_str)) {
        COMMON_LOG_ERROR << "Failed to get descriptor from message";
    }
}

void RecordProtoFactory::GetDescriptorString(const google::protobuf::Message& message, std::string* desc_str) {
    const google::protobuf::Descriptor* desc = message.GetDescriptor();
    return GetDescriptorString(desc, desc_str);
}

// Internal method
bool RecordProtoFactory::GetProtoDesc(const google::protobuf::FileDescriptor* file_desc, apollo::cyber::proto::ProtoDesc* proto_desc) {
    google::protobuf::FileDescriptorProto file_desc_proto;
    file_desc->CopyTo(&file_desc_proto);
    std::string str("");
    if (!file_desc_proto.SerializeToString(&str)) {
        return false;
    }

    proto_desc->set_desc(str);

    for (int i = 0; i < file_desc->dependency_count(); ++i) {
        auto desc = proto_desc->add_dependencies();
        if (!GetProtoDesc(file_desc->dependency(i), desc)) {
            return false;
        }
    }

    return true;
}

}  // namespace rosbag2_storage_record::internal