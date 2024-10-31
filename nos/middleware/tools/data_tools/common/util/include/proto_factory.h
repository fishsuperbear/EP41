#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <queue>
#include "ament_index_cpp/get_search_paths.hpp"
#include "google/protobuf/compiler/parser.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/io/tokenizer.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {

using namespace google::protobuf;

class ProtoFactory {
   public:
    ~ProtoFactory();

    static ProtoFactory* getInstance();

    // // Serialize all descriptors of the descriptor to string.
    static void GetDescriptorString(const google::protobuf::Descriptor* desc, std::string* desc_str);

    // Given a type name, constructs the default (prototype) Message of that type.
    // Returns nullptr if no such message exists.
    google::protobuf::Message* GenerateMessageByType(const std::string& type);

    // 删除拷贝构造函数和赋值运算符，防止外部复制实例
    ProtoFactory(const ProtoFactory&) = delete;
    ProtoFactory& operator=(const ProtoFactory&) = delete;

   private:
    bool init(bool open_proto_log);
    ProtoFactory();
    // bool RegisterMessage(const ProtoDesc& proto_desc);
    bool GetProtoFiles(const std::string folder_path, const std::string relative_path);
    google::protobuf::Message* GetMessageByGeneratedType(const std::string& type) const;
    std::queue<google::protobuf::FileDescriptorProto> file_desc_proto_vec_;

    std::unique_ptr<DescriptorPool> pool_ = nullptr;
    std::unique_ptr<DynamicMessageFactory> factory_ = nullptr;

    std::map<std::string, const google::protobuf::Message*> type_prototype_map_;

    static ProtoFactory* instance_;
    static std::mutex mutex_;

    // DECLARE_SINGLETON(ProtoFactory);
};

}  // namespace data_tool_common
}  // namespace netaos
}  // namespace hozon