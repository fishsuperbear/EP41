#include <time.h>
#include "adf-lite/include/base.h"
#include "adf-lite/service/rpc/lite_rpc.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
LiteRpc::LiteRpc(){

}

LiteRpc::~LiteRpc() {

}

void LiteRpc::RegisterServiceFunc(const std::string& func_name, ServiceFunc func){
    std::lock_guard<std::mutex> regist_lk(_regist_rpc_func_map_mutex);
    rpc_func_map[func_name] = func;
}

LiteRpc::ServiceFunc LiteRpc::GetServiceFunc(const std::string& func_name){
    std::lock_guard<std::mutex> regist_lk(_regist_rpc_func_map_mutex);
    if (rpc_func_map.find(func_name) == rpc_func_map.end()) {
        return nullptr;
    }

    return rpc_func_map[func_name];
}

void LiteRpc::RegisterStringServiceFunc(const std::string& func_name, StringServiceFunc func){
    std::lock_guard<std::mutex> regist_lk(_regist_rpc_string_func_map_mutex);
    rpc_string_func_map[func_name] = func;
}

LiteRpc::StringServiceFunc LiteRpc::GetStringServiceFunc(const std::string& func_name){
    std::lock_guard<std::mutex> regist_lk(_regist_rpc_string_func_map_mutex);
    if (rpc_string_func_map.find(func_name) == rpc_string_func_map.end()) {
        return nullptr;
    }

    return rpc_string_func_map[func_name];
}

}
}
}