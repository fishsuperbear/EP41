#pragma once

#include <functional>
#include <mutex>
#include <unordered_map>
#include "proto/common/header.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

class ProtoMessageTypeMgr {
   public:
    using CreateProtoMessageFunc = std::function<std::shared_ptr<google::protobuf::Message>()>;

    static ProtoMessageTypeMgr& GetInstance() {
        static ProtoMessageTypeMgr instance;

        return instance;
    }

    std::shared_ptr<google::protobuf::Message> Create(const std::string& name) {
        std::lock_guard<std::mutex> regist_lk(_regist_mutex);
        if (_func_map.find(name) != _func_map.end()) {
            return _func_map[name]();
        }

        return nullptr;
    }

    void Register(const std::string& name, CreateProtoMessageFunc func) {
        std::lock_guard<std::mutex> regist_lk(_regist_mutex);
        _func_map[name] = func;
    }

   private:
    ProtoMessageTypeMgr() {}

    std::unordered_map<std::string, CreateProtoMessageFunc> _func_map;
    std::mutex _regist_mutex;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon

#define REGISTER_PROTO_MESSAGE_TYPE(name, type)                                                                    \
    {                                                                                                              \
        hozon::netaos::adf::ProtoMessageTypeMgr::GetInstance().Register(                                           \
            name, []() { return std::static_pointer_cast<google::protobuf::Message>(std::make_shared<type>()); }); \
    }
