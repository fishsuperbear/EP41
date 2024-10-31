#include "proto_factory.h"
#include <dirent.h>
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include "ament_index_cpp/get_search_paths.hpp"
#include "data_tools_logger.hpp"

namespace hozon {
namespace netaos {
namespace data_tool_common {

// 初始化静态成员变量
ProtoFactory* ProtoFactory::instance_ = nullptr;
std::mutex ProtoFactory::mutex_;

ProtoFactory::ProtoFactory() {
    COMMON_LOG_DEBUG << "ProtoFactory init start.";
    pool_.reset(new DescriptorPool());
    factory_.reset(new DynamicMessageFactory(pool_.get()));
    init(false);
    COMMON_LOG_DEBUG << "ProtoFactory init end.";
}

ProtoFactory::~ProtoFactory() {
    factory_.reset();
    pool_.reset();
}

ProtoFactory* ProtoFactory::getInstance() {
    // static ProtoFactory instance;
    std::lock_guard<std::mutex> lock(mutex_);

    if (instance_ == nullptr) {
        instance_ = new ProtoFactory();
    }
    return instance_;
}

bool ProtoFactory::GetProtoFiles(const std::string folder_path, const std::string relative_path) {
    DIR* directory = opendir(folder_path.c_str());
    dirent* entry;
    while ((entry = readdir(directory)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        std::string fullPath = folder_path + "/" + entry->d_name;
        std::string proto_name = relative_path + "/" + entry->d_name;
        if (entry->d_type == DT_DIR) {
            GetProtoFiles(fullPath, proto_name);
        } else if (entry->d_type == DT_REG) {
            // 文件，检查文件名是否以 ".proto" 结尾
            std::string fileName(entry->d_name);
            if (fileName.size() >= 6 && fileName.substr(fileName.size() - 6) == ".proto") {
                //加载proto
                // 检查字符串是否非空且开头为斜杠符号
                if (!proto_name.empty() && proto_name[0] == '/') {
                    proto_name.erase(0, 1);
                }
                COMMON_LOG_DEBUG << "load proto file:  " << fullPath << "; name: " << proto_name;
                int fd1 = open(fullPath.c_str(), O_RDONLY);
                google::protobuf::io::FileInputStream text(fd1);
                google::protobuf::io::Tokenizer input(&text, NULL);
                google::protobuf::FileDescriptorProto file_desc_proto;
                google::protobuf::compiler::Parser parser;
                if (!parser.Parse(&input, &file_desc_proto)) {
                    COMMON_LOG_ERROR << "Failed to parse .proto :" << fullPath;
                    return false;
                }
                if (!file_desc_proto.has_name()) {
                    file_desc_proto.set_name(proto_name);
                }
                file_desc_proto_vec_.push(file_desc_proto);
            }
        }
    }
    closedir(directory);

    return true;
}

bool ProtoFactory::init(bool open_proto_log) {
    if (!open_proto_log) {
        //disable proto log
        google::protobuf::LogSilencer* logSilencer = new google::protobuf::LogSilencer();
        if (!logSilencer) {
            COMMON_LOG_WARN << "creat protobuf LogSilencer failed";
        }
    }

    DIR* dir;
    std::string folderPath = ament_index_cpp::get_search_paths().front() + "/proto";
    // load proto files
    dir = opendir(folderPath.c_str());
    if (dir == nullptr) {
        COMMON_LOG_DEBUG << "load proto files, but can't open folder " << folderPath;
        folderPath = "/app/conf/bag/proto";
        dir = opendir(folderPath.c_str());
        if (dir == nullptr) {
            COMMON_LOG_ERROR << "load proto files, but can't open folder " << folderPath;
            return false;
        }
    }
    closedir(dir);
    if (!GetProtoFiles(folderPath, "proto")) {
        return false;
    }
    while (!file_desc_proto_vec_.empty()) {
        google::protobuf::FileDescriptorProto file_desc = file_desc_proto_vec_.front();
        file_desc_proto_vec_.pop();
        if (!pool_->BuildFile(file_desc)) {
            file_desc_proto_vec_.push(file_desc);
        }
    }

    return true;
}

// Internal method
google::protobuf::Message* ProtoFactory::GenerateMessageByType(const std::string& type) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (type_prototype_map_.find(type) == type_prototype_map_.end()) {
        google::protobuf::Message* message = GetMessageByGeneratedType(type);
        if (message != nullptr) {
            return message;
        }

        const google::protobuf::Descriptor* descriptor = pool_->FindMessageTypeByName(type);
        if (descriptor == nullptr) {
            COMMON_LOG_ERROR << "cannot find [" << type << "] descriptor";
            return nullptr;
        }

        type_prototype_map_[type] = factory_->GetPrototype(descriptor);
        if (type_prototype_map_[type] == nullptr) {
            COMMON_LOG_ERROR << "cannot find [" << type << "] prototype";
            type_prototype_map_.erase(type);
            return nullptr;
        }
    }

    return type_prototype_map_[type]->New();
}

google::protobuf::Message* ProtoFactory::GetMessageByGeneratedType(const std::string& type) const {
    auto descriptor = pool_->FindMessageTypeByName(type);
    if (descriptor == nullptr) {
        COMMON_LOG_ERROR << "cannot find [" << type << "] descriptor";
        return nullptr;
    }

    auto prototype = factory_->GetPrototype(descriptor);
    if (prototype == nullptr) {
        COMMON_LOG_ERROR << "cannot find [" << type << "] prototype";
        return nullptr;
    }

    return prototype->New();
}

}  // namespace data_tool_common
}  // namespace netaos
}  // namespace hozon
