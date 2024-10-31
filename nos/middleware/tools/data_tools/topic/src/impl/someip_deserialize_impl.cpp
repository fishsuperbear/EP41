#include "someip_deserialize_impl.h"
#include "ara/com/serializer/someip_deserializer.h"
#include "ara/com/serializer/transformation.h"
#include "ara/core/result.h"
// #include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include "packet-someip.h"


namespace hozon {
namespace netaos {
namespace someip_deserialize {
SomeipDeserializeImpl* SomeipDeserializeImpl::instance_ = nullptr;
std::mutex SomeipDeserializeImpl::mutex_;

SomeipDeserializeImpl::SomeipDeserializeImpl(std::string json_format_path) {
    total_jsonformat_path_ = json_format_path + "TotalJsonFormat.json";
    ne_someip_serializer_api_ = std::make_unique<NESomeip::NESomeipSerializerAPI>();
    struct stat sb;
    if (stat(total_jsonformat_path_.c_str(), &sb) != 0) {
        std::string default_json_path = getEnvVar("SOMEIP_DESERIALIZ_JSON_PATH");
        total_jsonformat_path_ = default_json_path;
        if (stat(total_jsonformat_path_.c_str(), &sb) != 0) {
            std::cout << "someip TotalJsonFormat.json path: " << total_jsonformat_path_ << " not exist. can not echo someip payload! please use -c to specify TotalJsonFormat.json file path!" << std::endl;
        }
    }
    ne_someip_serializer_api_->loadFormatJson(total_jsonformat_path_);
}

SomeipDeserializeImpl::SomeipDeserializeImpl() {
    ne_someip_serializer_api_ = std::make_unique<NESomeip::NESomeipSerializerAPI>();
    struct stat sb;
    std::string default_json_path = getEnvVar("SOMEIP_DESERIALIZ_JSON_PATH");
    total_jsonformat_path_ = default_json_path;
    if (stat(total_jsonformat_path_.c_str(), &sb) != 0) {
        std::cout << "someip TotalJsonFormat.json path: " << total_jsonformat_path_ << " not exist. can not echo someip payload! " << std::endl;
    }
    ne_someip_serializer_api_->loadFormatJson(total_jsonformat_path_);
}

SomeipDeserializeImpl::~SomeipDeserializeImpl() {}

SomeipDeserializeImpl* SomeipDeserializeImpl::getInstance() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (instance_ == nullptr) {
        instance_ = new SomeipDeserializeImpl();
    }
    return instance_;
}

SomeipDeserializeImpl* SomeipDeserializeImpl::getInstance(std::string json_format_path = "./") {
    std::lock_guard<std::mutex> lock(mutex_);

    if (instance_ == nullptr) {
        instance_ = new SomeipDeserializeImpl(json_format_path);
    }

    return instance_;
}

std::string SomeipDeserializeImpl::deserialize(void* data, const int size, const std::string data_type) {
    // 解析要发送的数据
    unsigned char* p_message_data = reinterpret_cast<unsigned char*>(data);
    someip_message_t* p_someip_message = (someip_message_t*)p_message_data;
    someip_hdr_t someip_header = p_someip_message->someip_hdr;
    uint32_t someip_payload_length = p_someip_message->data_len;
    std::vector<uint8_t> payload(p_message_data + sizeof(someip_message_t), p_message_data + sizeof(someip_message_t) + someip_payload_length);

    string deserial_json;
    ne_someip_serializer_api_->deserializebyMethodID(
        someip_header.message_id.service_id,
        someip_header.message_id.method_id,
        someip_header.msg_type,
        payload,
        deserial_json);
    
    return deserial_json;
}

std::string SomeipDeserializeImpl::getEnvVar(const char* name) {
    char* value = getenv(name);
    if (value == nullptr) {
        std::cout << "error get env var: " << name;
        return "";
    }

    return std::string(value);
}

}  // namespace someip_deserialize
}  // namespace netaos
}  // namespace hozon