#pragma once
#include <iostream>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <json/json.h>
#include "NESomeipSerializerAPI.h"

namespace hozon {
namespace netaos {
namespace someip_deserialize {
class SomeipDeserializeImpl {
   public:
    SomeipDeserializeImpl();
    SomeipDeserializeImpl(std::string json_format_path);
    ~SomeipDeserializeImpl();

    static SomeipDeserializeImpl* getInstance();
    static SomeipDeserializeImpl* getInstance(std::string json_format_path);
    std::string deserialize(void* data, const int size, const std::string data_type);

   private:
    static SomeipDeserializeImpl* instance_;
    static std::mutex mutex_;
    std::string total_jsonformat_path_;
    std::unique_ptr<NESomeip::NESomeipSerializerAPI> ne_someip_serializer_api_;

    std::string getEnvVar(const char* name);
};

}  // namespace someip_deserialize

}  // namespace netaos
}  // namespace hozon
